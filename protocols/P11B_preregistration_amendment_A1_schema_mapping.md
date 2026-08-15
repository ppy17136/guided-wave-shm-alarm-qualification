# P11-B 预注册修订 A1：模式与轴映射

## 修订性质

本修订在 LOCK1 科学协议冻结后、任何 MAT 数值载入或波形统计之前作出。依据仅为 ZIP 中央目录、官方 README、官方 MATLAB 示例脚本以及 MAT 键名/形状元数据。它只解决文件路径、数组轴、循环映射和对齐窗口的执行细节，不改变主分数、6×MAD 阈值、温度邻居数、主要终点或成功门槛。

## 已审计事实

- ZIP CRC 通过；12,800 个文件，其中 12,795 个 MAT、3 个 M 脚本、2 个 TXT；
- 10,968 个 `niscope_avg_waveform.mat`，对应 1,828 个静态循环×6频率；
- 1,826 个 `niscope_waveforms.mat`，对应动态循环；
- 静态键：`niscopeAvgWaveform`，形状 `10000×4`；另有 `centerFreq`、`nAverages`、`numCycles`、`sampleRate`；
- 动态键：`niscopeWaveformsPlus/ZeroCrossing/Minus`，各 `10000×4`；
- 温度应变文件是 MATLAB 7.3/HDF5：`cyclingVectInterp` 为 `N×1`，`tc` 为 `1×N`，`dms` 为 `17×N`，N=12,132,621；
- 静态循环由官方脚本明确给出，首段从 420,000 开始；动态首段从 421,000 开始；
- 官方动态脚本把 MATLAB `80:610` 指定为首个到达波包窗口；
- 论文正文写疲劳激励约 14 Hz，而 `frequencies.txt` 写 124 Hz。P11-B 不使用该数值计算循环或信号特征，循环号完全采用官方文件路径/脚本，因此不受此差异影响；报告中保留该元数据矛盾。

## 固定执行映射

1. 静态路径模式为：`shaker_gw_sync_k/averaged/Loading_cycle_x/5_cycles_fkHz/niscope_avg_waveform.mat`；
2. 循环号直接解析 `Loading_cycle_x`，并与官方静态循环向量逐项核对；
3. 频率严格为 20、40、60、80、100、120 kHz；通道列 0–3 映射 S1–S4；
4. `niscopeAvgWaveform` 已是官方平均结果，不再读取或重做五次平均；
5. 对齐滞后只在 Python `[79:610)`（对应 MATLAB 80:610）窗口估计；
6. 得到滞后后，在完整 10,000 点的公共重叠区间计算残差，禁止用该窗口裁剪主残差；
7. 温度只读取 `cyclingVectInterp` 和 `tc`，不读取 `dms` 数值；
8. 温度数组按 HDF5 实际轴展平，要求长度相等且循环向量单调不降，否则正式运行中止；
9. 动态数据在主结果封存后才允许读取。

## 审计文件

- `data/reports/p11b_schema_audit_lock1/archive_schema_summary.json`
- `data/reports/p11b_schema_audit_lock1/archive_inventory.csv`
- `data/reports/p11b_schema_audit_lock1/mat_schema_summary.json`
- `data/reports/p11b_schema_audit_lock1/official_docs_only/`

本修订未查看任何波形、温度或应变数值，也未计算任何损伤分数。
