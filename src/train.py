from __future__ import annotations
import argparse,json,os,random
from pathlib import Path
import numpy as np, torch, yaml
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); p.add_argument("--experiment-id",required=True); p.add_argument("--method",required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--protocol",required=True); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
    config=yaml.safe_load(a.config.read_text(encoding="utf-8")); seed_all(a.seed); out=Path("runs")/a.experiment_id; out.mkdir(parents=True,exist_ok=True)
    prov={"experiment_id":a.experiment_id,"method":a.method,"seed":a.seed,"protocol":a.protocol,"config":config,"cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"pbs_jobid":os.environ.get("PBS_JOBID"),"pbs_array_index":os.environ.get("PBS_ARRAY_INDEX")}
    (out/"provenance.json").write_text(json.dumps(prov,ensure_ascii=False,indent=2),encoding="utf-8")
    if a.dry_run: print(json.dumps(prov,ensure_ascii=False,indent=2)); return
    raise SystemExit("Training is gated until pilot download, conversion, split audit and dry-run pass.")
if __name__=="__main__": main()

