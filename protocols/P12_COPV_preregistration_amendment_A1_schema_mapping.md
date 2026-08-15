# P12-COPV 预注册修订 A1：模式与目录映射

## 1. 修订性质

本修订发生在任何 `Data/Raw_Data` 数值读取、波形统计、异常分数、AUROC、FPR或损伤召回计算之前。三个ZIP已通过官方字节数与MD5，但A1只读取：

- ZIP中央目录；
- 18个官方 `H5_FILES_INFO.json`；
- 每个ZIP各一个代表H5的键名、形状、数据类型和属性名。

未读取原始波形数值。本修订只解决官方说明书与实际文件模式的命名/数量差异，不改变任何科学门槛、压力划分、频率、路径聚合、支持规则或主要终点。

## 2. 完整性结果

| 文件 | 官方字节数/MD5 | 本地核验 | 本地SHA-256 |
|---|---|---|---|
| `Baseline.zip` | 20,109,358,425 / `c57efcd712d82c9f735dce969176699e` | 通过 | `af210740a5e822e4d212e4462fd3b9eb6f07c318f1b56d7c99007f624581295a` |
| `Irreversible_Damage.zip` | 20,276,822,123 / `5bb46c4e92d5e873f2936e706001c366` | 通过 | `7b3e4b371bf15e1ae70d59e076c084650c09a976397de8d5020da003ee17d4ed` |
| `Reversible_Damage.zip` | 20,410,262,593 / `928e331d7b8fd4b3b2ca2189d57c5f79` | 通过 | `ef24744eafd38df4d8af25e91bc4437635f6a597d2ef473569cd9e18f6eda389` |

## 3. 档案结构

三个ZIP结构完全对称：

| 状态 | 成员 | 文件 | 目录 | H5 | JSON | TXT | FIG | 解压总字节 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 健康 | 210 | 198 | 12 | 180 | 6 | 6 | 6 | 117,802,427,196 |
| 不可逆 | 210 | 198 | 12 | 180 | 6 | 7 | 5 | 117,706,239,142 |
| 可逆 | 210 | 198 | 12 | 180 | 6 | 6 | 6 | 117,786,984,439 |

每个状态包含6个温度目录，每个温度恰好30个H5。

## 4. 压力序列修订

实际每个温度包含：

- 15个降压序列：`700,650,...,50,20 bar`；
- 15个随机序列：`100,650,700,300,150,350,500,600,250,550,20,450,400,50,200 bar`。

官方说明书文字称随机序列在50–700 bar之间，但中央目录显示随机序列也包含20 bar。三种结构状态和六个温度等级的顺序完全一致。

这不改变主分析：

- 主范围仍为50–700 bar；
- 校准仍使用随机序列 `{100,200,300,400,500,600,700}`；
- 未触碰健康参考测试仍使用 `{50,150,250,350,450,550,650}`；
- 两个集合各为7个压力×6温度×3重复；
- 20 bar仍为预声明次要外推分析。

## 5. H5实际键名

说明书以空格展示的数据集在H5中实际使用下划线：

| 说明书概念 | 实际H5路径 | 形状 | 类型 |
|---|---|---:|---|
| Raw Data | `Data/Raw_Data` | 18×600×7552 | float64 |
| Channel Inputs | `MetaData/Channel_Inputs` | 25×1 | float64 |
| Channels | `MetaData/Channels` | 600×2 | float64 |
| Chirp repetition index | `MetaData/Index_ChirpvsRepetition` | 1×3 | float64 |
| Frequency/repetition index | `MetaData/Index_FrequencyvsRepetition` | 2×15 | float64 |
| Pressure | `MetaData/Pressure` | 1×1 | float64 |
| Samples | `MetaData/Samples` | 1×1 | float64 |
| Sampling frequency | `MetaData/Sampling_Frequency` | 1×1 | float64 |
| Burst excitation | `MetaData/Signal_Data_Burst` | 5×23834 | float64 |
| Chirp excitation | `MetaData/Signal_Data_Chirp` | 1×31251 | float64 |
| Burst frequencies | `MetaData/Signal_Frequency_Burst` | 5×1 | float64 |
| Ambient temperature | `MetaData/Temp_Ambient` | 1×1 | float64 |
| Medium temperature | `MetaData/Temp_Medium` | 1×1 | float64 |
| Surface temperatures | `MetaData/Temp_Surface` | 4×1 | float64 |
| Timestamp | `Timestamp` | 6×1 | float64 |

三个代表H5均有相同17个对象和相同形状/类型。`MetaData/Channels`采用600×2而非说明书文字描述的2×600；执行代码按实际第二维的两列解释发送端与接收端，并在LOCK2前用合成结构自检确认。

## 6. 官方JSON映射

每个温度目录有一个 `H5_FILES_INFO.json`，均可解析为包含30项的列表。每项顶层键为：

- `folder`；
- `file`；
- `datasets`。

不可逆37°C JSON较小，与说明书所述外部温压日志缺失一致；主支持变量仍按预注册从H5元数据读取，缺失值不得事后插值。

## 7. 官方异常文件映射

说明书列出的四个健康异常文件均可在中央目录唯一定位。其中：

- T31 150 bar、T49 550 bar、T55 550 bar位于降压序列；
- T31 `25-06-03_09-27-02` 100 bar位于随机序列，因此该校准压力格减少一个H5块/三个重复；
- 理论每频率校准由126个分数降至123个，块数由42降至41，仍高于冻结的100样本和10块下限。

传感器20的全状态对称路径排除规则不变。

## 8. 存储与执行影响

每个ZIP解压约117.7 GB，三个同时完整解压约353.3 GB，不采用全量永久解压。执行代码必须：

1. 每次从ZIP流式抽取一个H5到受控临时目录；
2. 读取并处理该H5后立即删除；
3. 健康降压H5生成压缩float32模板库；
4. 随机健康与损伤H5只输出路径聚合后的小型特征表；
5. 临时空间至少预留一个H5的最大解压大小加安全余量；
6. 全过程记录成员CRC、路径、状态、温度、压力和序列类型。

## 9. A1结论

模式审计通过。三个档案的状态、温度、压力、重复、通道及核心H5结构高度对称；发现的随机序列20 bar和下划线键名属于纯模式映射，不改变主确认设计。

下一步允许实现并测试读取器、流式特征计算和执行编排，但在LOCK2建立前仍不得读取任何`Data/Raw_Data`数值。
