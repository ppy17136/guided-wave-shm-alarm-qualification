# P12-COPV 候选数据源审计与获取清单

## 1. 选择结论

P12 首选外部确认数据为 BAM 发布的 **Ultrasonic guided waves in a composite overwrapped pressure vessel under operational conditions**。该数据在结构、环境变量、工况变量和真实损伤方面同时满足双层可靠性门的验证需要：

- 新结构：额定 700 bar 的碳纤维缠绕复合压力容器；
- 健康基线：25–55 °C 六个温度等级；
- 工况：700–50 bar、每 50 bar 一档，另有 20 bar；同时包含降压顺序和随机压力顺序；
- 25 个永久粘接压电传感器，600 条有向激励—接收路径；
- 60、120、180、260、300 kHz 五个窄带激励和 60–300 kHz chirp；
- 每个温度—压力—激励组合三次重复；
- 可逆损伤：两个 200 g 铝块；
- 真实不可逆损伤：传感器 11、12 之间直径 8 mm、深度为壁厚 50% 的平底孔；
- H5 内包含压力、环境温度、液压介质温度和四点表面温度。

该组合能够直接检验 P12 的核心命题：阈值内部稳定并不等于目标温度—压力工况受支持，系统需要在域外工况明确拒判。

## 2. 官方记录

### 健康基线

- 页面：<https://zenodo.org/records/17776240>
- DOI：<https://doi.org/10.5281/zenodo.17776240>
- 访问状态：open
- 许可证：CC BY 4.0
- 发布日期：2026-01-16

### 损伤工况

- 页面：<https://zenodo.org/records/17782123>
- DOI：<https://doi.org/10.5281/zenodo.17782123>
- 访问状态：open
- 许可证：CC BY 4.0
- 发布日期：2026-01-16

## 3. 下载文件

建议存放目录：

`_references/P12_COPV_SEALED/`

### 第一批，完成主确认所必需

| 文件 | 字节数 | 官方 MD5 | 直接下载 |
|---|---:|---|---|
| `Baseline.zip` | 20,109,358,425 | `c57efcd712d82c9f735dce969176699e` | <https://zenodo.org/records/17776240/files/Baseline.zip?download=1> |
| `Irreversible_Damage.zip` | 20,276,822,123 | `5bb46c4e92d5e873f2936e706001c366` | <https://zenodo.org/records/17782123/files/Irreversible_Damage.zip?download=1> |

第一批合计 40,386,180,548 字节，约 40.39 GB（十进制）或 37.61 GiB。

### 第二批，次要分析

| 文件 | 字节数 | 官方 MD5 | 直接下载 |
|---|---:|---|---|
| `Reversible_Damage.zip` | 20,410,262,593 | `928e331d7b8fd4b3b2ca2189d57c5f79` | <https://zenodo.org/records/17782123/files/Reversible_Damage.zip?download=1> |

### 已下载说明书

- 文件：`_references/COPV_Repository_Documentation_v1.pdf`
- 字节数：20,442,614
- MD5：`9cf3ce371c583d2414a8d996c7176e4c`，与官方记录一致；
- SHA-256：`238b3a8ccd3d689759086620ff4f9cb33d572ec737caa667b413597df3bca77f`；
- PDF：7 页、A4、未加密，可完整提取文本并正常渲染。

## 4. 官方数据结构

每个 H5 的主要结构：

- `Data/Raw Data`：`18 × 600 × 7552`；
- 第一维：五个频率各三次重复以及 chirp 三次重复；
- 第二维：600 条有向发送—接收路径；
- 第三维：7552 点时间记录；
- `MetaData/Channels`：发送与接收传感器编号；
- `MetaData/Index FrequencyvsRepetition`：重复索引与频率；
- `MetaData/Index ChirpvsRepetition`：chirp 索引；
- `MetaData/Pressure`；
- `MetaData/Sampling Frequency`；
- `MetaData/Temp Ambient`、`Temp Medium`、`Temp Surface`。

每个温度目录还包括 H5 元数据 JSON、外部压力/温度 TXT 和 MATLAB 压力轨迹图。

## 5. 启封前必须冻结的官方异常处理

说明书已经公开列出以下问题，因此应在读取数值结果前固定处理规则：

1. 健康基线中四个文件存在全传感器 PSD 异常：
   - `T31 Baseline/25-06-03 09-00-20 GW Baseline T31 150bar.h5`；
   - `T31 Baseline/25-06-03 09-27-02 GW Baseline T31 100bar.h5`；
   - `T49 Baseline/25-06-04 16-26-17 GW Baseline T49 550bar.h5`；
   - `T55 Baseline/25-06-05 08-45-09 GW Baseline T55 550bar.h5`。
2. 不可逆损伤中传感器 20 在大量文件出现 PSD 异常。主协议将在所有状态中对称排除任何发送端或接收端为 20 的路径，不能只从损伤数据删除。
3. 不可逆损伤 37 °C 两个压力轨迹的外部温压日志缺失；T55 的 50 bar 和 200 bar 两个文件也缺失外部日志。
4. 主支持变量优先读取 H5 内部元数据。若上述文件的 H5 温压元数据也缺失，则样本标为 `unsupported/invalid`，不得用损伤标签指导插值。

## 6. 其他候选的定位

| 数据集 | 优点 | 不作为 P12 主确认的原因 |
|---|---|---|
| Open Guided Waves #2 | 20–60 °C、损伤、公开论文完整 | 已用于 P9 开发/迁移分析，不再未触碰 |
| 长期 Utah 导波数据 | 4.5 年环境变化、13 个事件 | 已用于 P1–P8，不能再次作为独立确认 |
| 风机叶片疲劳数据 | 温度、渐进损伤、静动态制度 | 已用于 P11B |
| 2026 单搭接接头 | 15.3 MB、多个裂纹长度 | 缺少系统环境变量，适合作为结构迁移补充 |
| 2026 水箱长期数据 | 2012–2020、1.4 GB | 官方页未说明独立损伤事件，适合健康误报审计 |
| MORPHO 50 kHz curated | 97.6 MB、五块复材板 | 主要标签为信号质量 GOOD/BAD，不是健康—损伤确认 |
| OGW #4 波场 | 完整脱粘波场 | 恒温、单次波场且约 6 GB，更适合损伤成像而非支持域报警 |

## 7. 密封规则

- 大型 ZIP 下载后只核对文件名、字节数、MD5 和 SHA-256；
- LOCK1 后允许列中央目录、读取官方 JSON 和一个 H5 的键名/形状/数据类型，但不得统计波形值；
- 完成模式映射 A1、读取器、自检和执行代码后建立 LOCK2；
- LOCK2 前不得计算任何健康或损伤异常分数；
- 正式执行一次完成，结果无论通过、失败或高拒判率均保留。

## 8. 获取建议

由于单文件约 20 GB，建议使用支持断点续传的下载器手工下载到 E 盘。主确认只需先下载 `Baseline.zip` 和 `Irreversible_Damage.zip`；`Reversible_Damage.zip` 可在主结果完成后作为预声明次要分析下载。下载完成后不要解压，直接通知文件已到位即可。
