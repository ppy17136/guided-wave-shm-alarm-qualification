"""Train/evaluate the P2 dual-branch SSL model on causal D1--D6 event windows."""
from __future__ import annotations

import argparse, csv, json, os, random, traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
import zarr
from torch.utils.data import DataLoader, Dataset

from src.envwave.event_dual_branch import EventDualBranchSSL, variance_covariance_regularizer
from src.envwave.model import masked_wave
from src.envwave.numpy_metrics_v1 import average_precision, roc_auc


ROOT = Path(__file__).resolve().parents[1]
DAY_NS = 86_400_000_000_000


def now(): return datetime.now(timezone.utc).isoformat()
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"); tmp.replace(path)


def environment_raw(group, indices, times):
    temp = np.asarray(group["temperature"].oindex[indices], np.float64)
    hours = times.astype(np.float64) / 3_600_000_000_000
    return np.column_stack([temp, temp * temp, np.asarray(group["humidity"].oindex[indices], np.float64),
        np.log1p(np.maximum(np.asarray(group["brightness"].oindex[indices], np.float64), 0)),
        np.asarray(group["pressure"].oindex[indices], np.float64), np.sin(2*np.pi*hours/24), np.cos(2*np.pi*hours/24)])


def build_records(start_ns, stop_ns):
    rows = []
    for store in sorted((ROOT / "data/zarr").glob("measurements_2021_*.zarr")):
        group = zarr.open_group(str(store), mode="r"); times = np.asarray(group["datetime_ns"][:], np.int64)
        for idx in np.flatnonzero((times >= start_ns) & (times < stop_ns)):
            rows.append((int(times[idx]), str(store), int(idx)))
    rows.sort(); return rows


def select(records, start, stop): return [r for r in records if start <= r[0] < stop]


def feature_stats(records):
    amps, envs = [], []
    by_store = {}
    for time, store, idx in records: by_store.setdefault(store, []).append((time, idx))
    for store, items in by_store.items():
        group = zarr.open_group(store, mode="r"); idx = np.asarray([x[1] for x in items], np.int64); times = np.asarray([x[0] for x in items], np.int64)
        for start in range(0, len(idx), 256):
            batch_idx = idx[start:start+256]; wave = np.asarray(group["guided_wave"].oindex[batch_idx, :, :], np.float32)
            amps.append(np.log(np.sqrt(np.mean(wave.astype(np.float64)**2, axis=2)) + 1e-12))
        envs.append(environment_raw(group, idx, times))
    amp = np.concatenate(amps); env = np.concatenate(envs)
    return amp.mean(0), np.maximum(amp.std(0), 1e-6), env.mean(0), np.maximum(env.std(0), 1e-6)


class EventDataset(Dataset):
    def __init__(self, records, stats): self.records, self.stats, self.groups = records, stats, {}
    def __len__(self): return len(self.records)
    def __getitem__(self, item):
        time, store, idx = self.records[item]
        if store not in self.groups: self.groups[store] = zarr.open_group(store, mode="r")
        group = self.groups[store]; wave = np.asarray(group["guided_wave"][idx], np.float32)
        rms = np.sqrt(np.mean(wave.astype(np.float64)**2, axis=1)); shape = wave - wave.mean(1, keepdims=True)
        shape /= np.sqrt(np.mean(shape.astype(np.float64)**2, axis=1, keepdims=True)).astype(np.float32) + 1e-6
        env = environment_raw(group, np.asarray([idx]), np.asarray([time]))[0]
        am, ast, em, est = self.stats
        return {"shape": torch.from_numpy(shape), "amp": torch.from_numpy(((np.log(rms+1e-12)-am)/ast).astype(np.float32)),
            "env": torch.from_numpy(((env-em)/est).astype(np.float32)), "time": torch.tensor(time, dtype=torch.int64)}


def loader(records, stats, batch, workers, shuffle):
    return DataLoader(EventDataset(records, stats), batch_size=batch, shuffle=shuffle, num_workers=workers,
        pin_memory=torch.cuda.is_available(), persistent_workers=workers>0, drop_last=False)


def ssl_loss(model, shape, amp, env, ratio, block, weights):
    corrupted, mask = masked_wave(shape, ratio, block); out = model(corrupted, amp, env)
    shape_loss = (torch.square(out["wave_reconstruction"]-shape)*mask).sum()/mask.sum().clamp_min(1)
    amp_loss = torch.square(out["amplitude_prediction"]-amp).mean()
    corrupted2, _ = masked_wave(shape, ratio, block); out2 = model(corrupted2, amp, env)
    consistency = 1-torch.nn.functional.cosine_similarity(out["z_fused"], out2["z_fused"], dim=1).mean()
    regularizer = variance_covariance_regularizer(out["z_fused"])
    loss = weights[0]*shape_loss + weights[1]*amp_loss + weights[2]*consistency + weights[3]*regularizer
    return loss, [shape_loss, amp_loss, consistency, regularizer]


@torch.no_grad()
def extract(model, data_loader, device, ratio, block, seed):
    model.eval(); seed_all(seed); result={k:[] for k in ("embedding","shape_error","amp_error","time")}
    for batch in data_loader:
        shape=batch["shape"].to(device); amp=batch["amp"].to(device); env=batch["env"].to(device)
        corrupted, mask=masked_wave(shape, ratio, block); out=model(corrupted, amp, env)
        se=(torch.square(out["wave_reconstruction"]-shape)*mask).sum((1,2))/mask.sum((1,2)).clamp_min(1)
        ae=torch.square(out["amplitude_prediction"]-amp).mean(1)
        result["embedding"].append(out["z_fused"].cpu().numpy()); result["shape_error"].append(se.cpu().numpy())
        result["amp_error"].append(ae.cpu().numpy()); result["time"].append(batch["time"].numpy())
    return {k:np.concatenate(v) for k,v in result.items()}


def score_fit(train, *sets):
    emb=train["embedding"]; center=emb.mean(0); cov=np.cov(emb,rowvar=False); d=cov.shape[0]
    cov=.85*cov+.15*np.eye(d)*np.trace(cov)/d; inv=np.linalg.pinv(cov,rcond=1e-6)
    def raw(x):
        delta=x["embedding"]-center; md=np.einsum("ni,ij,nj->n",delta,inv,delta)
        return np.column_stack([np.log1p(md), np.log1p(x["shape_error"]), np.log1p(x["amp_error"])])
    tr=raw(train); med=np.median(tr,0); q25,q75=np.quantile(tr,[.25,.75],axis=0); scale=np.maximum(q75-q25,1e-6)
    return [np.maximum((raw(x)-med)/scale,0)@np.asarray([1.0,.5,.5]) for x in sets]


def runs(mask,k=3):
    p=np.r_[False,mask,False]; e=np.flatnonzero(p[1:]!=p[:-1]); return int(np.sum((e[1::2]-e[::2])>=k))


def metrics(cal_score, pre_score, post_score, pre_time, post_time, event_ns, q):
    threshold=float(np.quantile(cal_score,q,method="higher")); pa=pre_score>threshold; oa=post_score>threshold
    conv=np.convolve(oa.astype(np.int8),np.ones(3,dtype=np.int8),mode="valid"); found=np.flatnonzero(conv==3)
    delay=None if not len(found) else float((post_time[int(found[0])]-event_ns)/3_600_000_000_000)
    labels=np.r_[np.zeros(len(pre_score),np.int8),np.ones(len(post_score),np.int8)]; scores=np.r_[pre_score,post_score]
    return {"threshold":threshold,"pre_fpr":float(pa.mean()),"pre_false_runs_k3":runs(pa),"post_recall":float(oa.mean()),
        "detection_delay_hours_k3":delay,"roc_auc":roc_auc(labels,scores),"average_precision":average_precision(labels,scores)}


def run_seed(config,event,seed,smoke,force):
    event_ns=int(np.datetime64(event["event_time_utc"].removesuffix("Z"),"ns").astype(np.int64)); eid=int(event["event_index"])
    base_output = config["runtime"].get("smoke_output_dir", "runs/p2_event_ssl_v1_smoke/01") if smoke else config["runtime"]["output_dir"]
    out=ROOT/base_output/f"event_{eid:02d}"/f"seed_{seed}"; done=out/"COMPLETED.json"
    if done.exists() and not force: return {"event":eid,"seed":seed,"status":"skipped"}
    out.mkdir(parents=True,exist_ok=True); atomic_json(out/"RUNNING.json",{"started_utc":now(),"event":eid,"seed":seed})
    try:
        records=build_records(event_ns-15*DAY_NS,event_ns+5*DAY_NS)
        tr=select(records,event_ns-15*DAY_NS,event_ns-8*DAY_NS); cal=select(records,event_ns-8*DAY_NS,event_ns-5*DAY_NS)
        pre=select(records,event_ns-5*DAY_NS,event_ns); post=select(records,event_ns,event_ns+5*DAY_NS)
        if smoke: tr=tr[::max(1,len(tr)//256)][:256]; cal=cal[::max(1,len(cal)//128)][:128]; pre=pre[::max(1,len(pre)//128)][:128]; post=post[::max(1,len(post)//128)][:128]
        stats=feature_stats(tr); tc=config["training"]; mc=config["model"]; batch=16 if smoke else int(tc["batch_size"]); workers=0 if smoke else int(tc["workers"])
        loaders=[loader(x,stats,batch,workers,i==0) for i,x in enumerate((tr,cal,pre,post))]
        seed_all(seed); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model=EventDualBranchSSL(int(mc["embedding_dim"]),7,int(mc["transformer_layers"]),int(mc["attention_heads"]),float(mc["dropout"])).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=float(tc["learning_rate"]),weight_decay=float(tc["weight_decay"])); use_amp=device.type=="cuda" and bool(tc["mixed_precision"])
        scaler=torch.amp.GradScaler("cuda",enabled=use_amp); epochs=2 if smoke else int(tc["epochs"]); best=float("inf"); state=None; stale=0; history=[]
        weights=[float(x) for x in tc["loss_weights"]]
        for epoch in range(1,epochs+1):
            model.train(); losses=[]
            for b in loaders[0]:
                shape=b["shape"].to(device); amp=b["amp"].to(device); env=b["env"].to(device); opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp): loss,parts=ssl_loss(model,shape,amp,env,float(mc["mask_ratio"]),int(mc["mask_block_samples"]),weights)
                scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update(); losses.append(float(loss.detach().cpu()))
            cal_out=extract(model,loaders[1],device,float(mc["mask_ratio"]),int(mc["mask_block_samples"]),seed+epoch); val=float(np.mean(cal_out["shape_error"]+cal_out["amp_error"]))
            history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"calibration_loss":val})
            if val<best: best=val; state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; stale=0
            else: stale+=1
            if stale>=(1 if smoke else int(tc["patience"])): break
        model.load_state_dict(state); model.to(device); outs=[extract(model,l,device,float(mc["mask_ratio"]),int(mc["mask_block_samples"]),seed+1000+i) for i,l in enumerate(loaders)]
        cal_score,pre_score,post_score=score_fit(outs[0],outs[1],outs[2],outs[3]); result=metrics(cal_score,pre_score,post_score,outs[2]["time"],outs[3]["time"],event_ns,float(config["evaluation"]["threshold_quantile"]))
        result.update({"event_index":eid,"transition":event["transition"],"seed":seed,"epochs":len(history),"device":str(device),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        atomic_json(out/"metrics.json",result); torch.save({"model_state":state,"stats":[x.tolist() for x in stats],"config":config},out/"checkpoint.pt")
        atomic_json(out/"provenance.json",{"schema_version":"p2-event-ssl-provenance-v1","completed_utc":now(),
            "event":event,"seed":seed,"smoke":smoke,"split_counts":{"train":len(tr),"calibration":len(cal),"pre_event":len(pre),"post_event":len(post)},
            "causal_split":"train[-15d,-8d), calibration[-8d,-5d), pre[-5d,0), post[0,+5d)",
            "leakage_guard":"post-event samples are excluded from training, model selection and threshold calibration",
            "versions":{"python":os.sys.version,"numpy":np.__version__,"torch":torch.__version__,"zarr":zarr.__version__},
            "pbs_jobid":os.environ.get("PBS_JOBID"),"host":os.environ.get("HOSTNAME")})
        with (out/"history.csv").open("w",newline="",encoding="utf-8-sig") as f: w=csv.DictWriter(f,fieldnames=history[0].keys());w.writeheader();w.writerows(history)
        atomic_json(done,{"completed_utc":now(),"event":eid,"seed":seed}); (out/"RUNNING.json").unlink(missing_ok=True); return result
    except Exception as exc:
        atomic_json(out/"FAILED.json",{"failed_utc":now(),"error":str(exc),"traceback":traceback.format_exc()}); (out/"RUNNING.json").unlink(missing_ok=True); raise


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,required=True);p.add_argument("--dry-run",action="store_true");p.add_argument("--execute",action="store_true");p.add_argument("--smoke",action="store_true");p.add_argument("--force",action="store_true");a=p.parse_args()
    config=yaml.safe_load((ROOT/a.config).read_text(encoding="utf-8")); manifest=json.loads((ROOT/config["data"]["event_manifest"]).read_text(encoding="utf-8")); events=manifest["events"]
    audit={"events":len(events),"transitions":[e["transition"] for e in events],"stores":sorted(p.name for p in (ROOT/"data/zarr").glob("measurements_2021_*.zarr")),"cuda":torch.cuda.is_available(),"config":str(a.config)}
    print(json.dumps(audit,ensure_ascii=False,indent=2))
    if a.dry_run or not a.execute:
        if not a.execute:return
    if a.execute and not torch.cuda.is_available() and not a.smoke: raise SystemExit("Full execution requires a visible CUDA GPU")
    selected=events[:1] if a.smoke else events; seeds=config["training"]["seeds"][:1] if a.smoke else config["training"]["seeds"]
    results=[run_seed(config,e,int(s),a.smoke,a.force) for e in selected for s in seeds]
    print(json.dumps({"results":results},ensure_ascii=False,indent=2))


if __name__=="__main__":main()
