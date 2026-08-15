# P7 预注册修正案 A1：P5b 慢参考更新 guard

修正时间：2026-08-10（在 2022_05–2022_10 数据完成下载、转换或任何导波幅值/模型分数计算之前）

## 修正内容

`P7_D7_D13_confirmatory_preregistration_v1.md` 第 5 节与 `P7-CONFIRM-D7-D13.yaml` 将 P5b 的慢参考更新条件误写为 `p > 0.20`。经回查已经在 D1–D6 上运行并产生锁定结果的源代码 `tools/analyze_p5b_leaky_cusum.py`，其冻结实现为：

```python
GUARD_P = .05
if not alarm and p_value > GUARD_P and accepted % slow_stride == 0:
    slow.append(float(raw_score))
```

因此，P7 确认实验必须使用 **`p > 0.05`**，才能忠实复现被预注册的 P5b calstable 算法。其他参数保持不变：fast window 128、slow stride 32、κ=4、h=8、decay=0.95、K=3。

## 修正依据与防止结果驱动声明

1. 修正依据仅为已冻结且哈希已记录的 D1–D6 源代码，不是 D7–D13 结果；
2. 修正时未来文件仅处于传输阶段，尚未进行 pickle 解析、Zarr 转换、幅值统计、模型训练、重评分或标签分层绘图；
3. 不改变主方法、主终点、成功标准、比较方法、随机种子或统计检验；
4. 原 v1 文件和 SHA-256 封印永久保留，本修正案与 v1.1 配置共同构成 P7 的最终执行规范。

