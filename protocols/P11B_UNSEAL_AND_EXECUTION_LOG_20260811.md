# P11-B 启封与一次性执行日志

## 数据身份

- 源 ZIP：`_references/wind_turbine_blade_shm_dataset.zip`
- 文件大小：1,695,645,067 bytes
- SHA-256：`02394f981f2d9dc757dfe194d321bc8383ccb8d806bde009e60e23547a7c5e5e`
- 与此前 SEALED 副本字节级一致。

## LOCK1

- 科学协议在任何 ZIP 内部目录、README、MAT 模式或数值查看前冻结；
- LOCK1 清单 SHA-256：`7bec40dec4d5b4ea8023b8c63c2df3fe680043e9a3aa476dfcc0ea63f2606d5b`；
- LOCK1 后只进行了中央目录、官方文本/脚本和 MAT 键名/形状审计；未读取数值。

## LOCK2

- 执行脚本先在合成信号上通过自检；
- 执行脚本 SHA-256：`9a287c45ae2e7c650ed9fa288f3a569454e4cfb671f3d8b57c019496ffcfc0ac`；
- LOCK2 清单 SHA-256：`f00bcb6de8480743ca55e024f3ed3ed8145c4a8ab985eb2e6fae1c45bfa42781`；
- LOCK2 之后未修改主脚本、阈值、边界、通道聚合或终点。

## 正式启封与执行

- 本地开始时间：2026-08-11 18:26:47 +08:00；
- 本地完成时间：2026-08-11 18:29:14 +08:00；
- 命令：`python tools/run_p11b_wind_blade_confirmatory.py --execute`；
- 退出码：0；
- 正式数值执行次数：1；
- 静态波形：1,828 个循环、6 个频率、4 个传感器、每波形 10,000 点；
- 正式状态：完成；
- 主判定：FAIL（校准可信度门失败）；
- 未进行任何阈值重算、温度边界调整或主结果重跑。

## 结果位置

`runs/p11b_wind_blade_confirmatory_v1/`

主结果封存后进行的所有进一步统计均放入 `posthoc_diagnostics/`，并标注为事后描述性分析。
