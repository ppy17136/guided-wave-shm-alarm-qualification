# P7 D7–D13 数据获取与转换指令

预注册文件与冻结清单生成后，才运行以下步骤。

## 本地 E 盘下载（约 9.54 GiB）

```powershell
cd '<PROJECT_ROOT>'
python tools\download_figshare.py --months 2022_05 2022_06 2022_07 2022_08 2022_09 2022_10 --yes
```

脚本支持 `.part` 断点续传，并逐文件核对 Figshare 官方 MD5。文件写入 `data\raw`。

## 转换为 Zarr v2 并重建事件元数据

建议在本地现有 `.venv_data` 环境执行：

```powershell
cd '<PROJECT_ROOT>'
.\.venv_data\Scripts\python.exe tools\prepare_verified_event_months.py 2022_05 2022_06 2022_07 2022_08 2022_09 2022_10
```

完成后必须先只检查事件清单，不运行任何分数分析：

```powershell
.\.venv_data\Scripts\python.exe -c "import json,pathlib; p=pathlib.Path('data/reports/damage_event_protocol_v1/damage_event_manifest.json'); d=json.loads(p.read_text(encoding='utf-8')); print([(x['transition'],x['event_time_utc'],x['baseline_complete'],x['pre_event_complete'],x['post_event_complete']) for x in d['events']])"
```

预期总事件为 14 个（含 D1→D6、D1→D0 恢复和 D6→D13）；P7 执行器只选择 `old_tag>=6 and new_tag==old_tag+1` 的 7 个确认事件。

## 上传集群

不要再次上传约 9.54 GiB pickle。只上传转换后的 6 个 Zarr 目录、P7 执行包及校验清单至：

`<PROJECT_ROOT>`

集群端在提交唯一 PBS 任务前，必须完成 7 事件、35 checkpoint 计划、数据形状、CUDA 模型前向和冻结哈希预检。


