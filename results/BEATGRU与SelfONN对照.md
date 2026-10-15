# BEATGRU 与 Self-ONN：性能与迁移性对照

## 可以直接比较的部分：MIT-BIH DS1/DS2

两者均使用公开 MIT-BIH DS1/DS2 列表和 N/S/V pooled 计算；Self-ONN 数值来自本地公开代码严格复现，BEATGRU 数值来自发布 seed-0 checkpoint。

| 方法 | 输入因果性 | Macro-F1 | N-F1 | S-F1 | V-F1 | SV-F1 |
|---|---|---:|---:|---:|---:|---:|
| Self-ONN 公开代码复现 | 离线：后 RR、整条记录 RR 均值、整条记录 CWT | 0.895782 | 0.986951 | 0.745168 | **0.955229** | 0.850198 |
| BEATGRU | 严格因果：`R_i+200 ms`、历史 RR、记录内 GRU 状态 | **0.913130** | **0.988864** | **0.802646** | 0.947879 | **0.875262** |
| BEATGRU − Self-ONN | — | +0.017347 | +0.001913 | +0.057478 | −0.007349 | +0.025064 |

结论：在该公开 DS1/DS2 划分、同一 N/S/V F1 汇总方式下，BEATGRU 的 Macro-F1 和 S-F1 更高；Self-ONN 的 V-F1 高约 0.00735。尤其应注意，BEATGRU 的比较优势不是通过使用后 RR 或整条测试记录统计得到的。

## 外部迁移：当前只有 BEATGRU 的有效证据

已核对原论文 *Global ECG Classification by Self-Operational Neural Networks With Feature Injection*（IEEE TBME 2023，DOI: `10.1109/TBME.2022.3187874`）：其 Results 明确以 MIT-BIH Arrhythmia benchmark 为实验数据库。作者的公开仓库也明确写为 “tested on the MIT-BIH Arrhythmia dataset”。论文与仓库均未报告 MIT-BIH→SVDB 或 MIT-BIH→INCART 的完整冻结 zero-shot 实验。

| 源域训练 | 目标完整数据库 | 方法 | 设置 | Macro-F1 | S-F1 | V-F1 |
|---|---|---|---|---:|---:|---:|
| MIT-BIH DS1 | SVDB（78 条记录） | BEATGRU | 完全冻结 zero-shot | 0.569413 | 0.272495 | 0.509362 |
| MIT-BIH DS1 | INCART（75 条记录） | BEATGRU | 完全冻结 zero-shot，canonical-first/I | 0.728193 | 0.444548 | 0.760235 |
| MIT-BIH DS1 | INCART（75 条记录） | BEATGRU | 完全冻结 zero-shot，预先声明的 Lead-II 匹配敏感性分析 | 0.814848 | 0.597697 | 0.864191 |
| MIT-BIH DS1 | SVDB / INCART | Self-ONN | **无本地或公开同协议结果** | — | — | — |

因此不能写成“BEATGRU 的跨域泛化优于 Self-ONN”；正确表述是：

> BEATGRU 已在完整 SVDB 和 INCART 上给出冻结 zero-shot 迁移证据，而当前复现的 Self-ONN 只验证了 MIT-BIH DS1/DS2，尚无同协议跨库结果。

## 为什么不能直接假定 Self-ONN 的迁移结果

本地审计的 Self-ONN 公开实现：

1. 数据划分和预处理针对 MIT-BIH DS1/DS2 固定，原始流程不是“训练一个模型后接收任意 WFDB 数据库”的通用评估入口；
2. 中值滤波窗口按 360 Hz 固定，SVDB 为 128 Hz、INCART 为 257 Hz，直接复用会改变滤波的物理时间尺度；
3. 当前预测使用下一 R 峰构造后 RR，以及整条测试记录的平均 RR；
4. 对整条记录先计算 CWT，再切取心拍图块，属于离线记录级变换；
5. 它没有提交可直接用于 SVDB/INCART 的训练 checkpoint 或完整迁移评估脚本。

这并不表示 Self-ONN 必然迁移更差；只是当前没有合法数值支持这一判断。若要得到严格的横向迁移结论，应固定源域 DS1 训练集、将 Self-ONN 重实现为可处理 128/257 Hz 的统一预处理，并在 SVDB/INCART 全部记录上冻结测试。那将是“离线 Self-ONN zero-shot”对照，不应和 BEATGRU 的严格因果部署口径混为同一个主表。

## 对 BEATGRU 迁移结果的评价

- **优点**：跨域评估覆盖完整目标数据库、没有目标域训练或调参，并保留严格因果状态管理；这是真实而可复核的域外证据。
- **不足**：SVDB 的 S/V 明显下降，说明当前 MIT-BIH 单域训练未学习到充分的设备、采样率、导联和节律分布不变性。
- **INCART 的发现**：物理 Lead-II 一致性显著改善迁移，说明导联语义匹配比“取第一个输入通道”更重要。这是后续改进导联不变编码器的直接证据。

论文上最稳妥的定位是：BEATGRU 在严格因果 DS1/DS2 上优于当前复现的 Self-ONN，并提供了 Self-ONN 尚未提供的完整跨库零样本评估；但外部迁移尤其在 SVDB 上仍是当前方法的主要限制。
