# BEATGRU：严格因果在线心拍分类

这是可独立发送的 MIT-BIH Arrhythmia N/S/V 心拍分类代码。它实现当前最佳方法：**raw+d1+d2 心拍 CNN + frozen-logit causal GRU**，使用标准 MIT-BIH **inter-patient DS1/DS2** 划分。

项目不包含 MIT-BIH 原始波形。请从 PhysioNet 获得 `mit-bih-arrhythmia-database-1.0.0`，并遵守该数据集的使用条款。

## 结果

在标准 DS1/DS2、N/S/V pooled 指标下，最佳 seed 的测试结果为：

| Macro-F1 | N-F1 | S-F1 | V-F1 | SV-F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| **0.913130** | 0.988864 | 0.802646 | 0.947879 | 0.875262 | 0.979072 |

固定同一 CNN backbone 的 seed 1–3：Macro-F1 `0.912972 ± 0.001292`。完整的逐 seed CSV、混淆矩阵和公开 Self-ONN 对照在 [results](results/README.md)。

这些数值与离线方法比较时，应仅和同样采用 DS1/DS2、AAMI N/S/V 定义及 pooled F1 计算的工作直接比较。

## 严格因果协议

对第 `i` 个心拍，在 `R_i + 200 ms` 输出类别，条件于该 R 峰已经被检测/标注。允许的输入仅包括：

- `R_i-350 ms` 至 `R_i+200 ms` 的当前波形；
- 当前与历史 R 峰导出的 RR 特征；
- 同一记录中先前心拍更新出的 GRU hidden state。

禁止输入下一心拍、post-RR、`token_rr_next`、未来 R 峰、未来标签、测试记录全局统计量和双向时序模型。CNN 的一阶和二阶差分只在上述合法波形窗口中计算。GRU 在每条记录开始时重置，测试阶段从不接收真实历史类别。

该任务是“**conditional on detected/annotated R-peaks**”的在线心拍分类；不将 R 峰检测本身宣称为端到端实时任务。

## 安装

建议 Python 3.9+ 与 CUDA PyTorch 环境：

```bash
git clone https://github.com/hanzhaoheng666/ecg_BEATGRU.git BEATGRU
cd BEATGRU
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

如需 GPU，请根据本机 CUDA 安装匹配版本的 PyTorch，再安装其余依赖。

## 数据与一键复现

假定原始数据目录含有 `100.hea`、`100.dat`、`100.atr` 等 48 条 MIT-BIH 记录：

```bash
bash scripts/build_manifest.sh /path/to/mit-bih-arrhythmia-database-1.0.0
GPUS=0,1,2 bash scripts/train_3gpu.sh data/mitbih_ds1ds2
```

第二步先在第一张卡训练一次 3 通道 derivative CNN，再在三张卡并行训练和测试 GRU seed 0、1、2。输出写入新的 `runs/causal_ds1ds2_*/` 目录：

```text
runs/.../
├── backbone/derivative_cnn.pt
├── gru_seed0/beatgru.pt
├── gru_seed0/test/test_metrics.json
├── gru_seed0/test/test_predictions.npz
├── gru_seed1/ ...
└── gru_seed2/ ...
```

汇总结果与执行审计：

```bash
bash scripts/summarize.sh runs/causal_ds1ds2_YYYYMMDD_HHMMSS
bash scripts/audit.sh data/mitbih_ds1ds2 \
  runs/causal_ds1ds2_YYYYMMDD_HHMMSS/gru_seed0/beatgru.pt
```

审计会检查未来波形不变性、未来 RR/标签不变性、前缀 replay 一致性与记录状态重置。GPU 不可用时，使用 `DEVICE=cpu bash scripts/train_3gpu.sh ...` 和 `DEVICE=cpu bash scripts/audit.sh ...`。

## 发布 checkpoint 的直接评估

`checkpoints/` 已包含最优 seed 0 的 CNN 与 GRU 权重（约 1 MB，总 SHA256 在 `checkpoints/SHA256SUMS`）。在构建 manifest 后，无需重新训练即可评估：

```bash
bash scripts/evaluate_release_checkpoint.sh data/mitbih_ds1ds2
bash scripts/audit.sh data/mitbih_ds1ds2 checkpoints/beatgru_best_seed0.pt \
  runs/release_seed0_audit.json checkpoints/derivative_cnn_seed0.pt
```

## 外部泛化测试

代码支持将完全冻结的 MIT-BIH checkpoint 直接评估到 SVDB/INCART 的**全部记录**。目标域不应参与任何选择或微调：

```bash
GPUS=0,1 bash scripts/evaluate_external_zero_shot.sh \
  /path/to/v4_strict_causal_external_manifest
```

该 manifest 应含 `mit_bih_svdb.csv` 与 `st_petersburg_incart.csv`，并且 CSV 的事件顺序、R 峰和 N/S/V 评分规则必须经过审计。当前已完成的外部泛化结果见 [results/2026-09-18_外部泛化性测试.md](results/2026-09-18_外部泛化性测试.md)。

外部推理的导联规则必须在查看任何目标标签前固定：发布模型的源域输入为 MIT-BIH MLII，因此脚本对具有明确 II 名称的 INCART 固定使用 II；SVDB 的两个通道仅命名为 ECG1/ECG2、没有可验证的物理导联语义，故固定使用 ECG1，不能以目标域指标选择通道。

## DS1 六折选模与 DS2 最终一次评估

`experiments/ds1_6fold_causal/candidates.json` 预先固定三组 GRU 候选。下列独立脚本只在 DS1 记录中构建六个互斥 fold、选择单一候选，然后以完整 DS1 重训并只加载一次 DS2：

```bash
GPUS=0,1,2 bash scripts/run_ds1_6fold_then_ds2_once.sh \
  data/mitbih_ds1ds2
```

输出的 `LOCKED_CONFIG.json` 保存所有 DS1 验证结果与唯一入选配置；`final_ds1_retrain/.../DS2_FINAL_ONCE/test_metrics.json` 是后续唯一的 DS2 测试输出。该流程不搜索类别阈值、logit 偏置或目标域参数。

## 双导联与形态历史候选（V / 跨域）

`scripts/run_pair_embedding_6fold_and_frozen.sh` 是与发布单导联模型隔离的候选实验。它在 DS1 记录六折中比较固定双导联、导联顺序交换和形态 embedding 历史三种预定义方案；选出唯一候选后，以全 DS1 重训、仅一次评估 DS2，并冻结测试全部 INCART 与 SVDB 记录：

```bash
GPUS=0,1,2 EPOCHS=15 bash scripts/run_pair_embedding_6fold_and_frozen.sh \
  /path/to/raw_mitbih_onn_ds1ds2 \
  /path/to/manifest/strict_online_all_rpeaks
```

导联规则在测试前固定：MIT-BIH/INCART 为具名 II 加首个可用胸前导联；SVDB 无 II 名称，固定读取其 `ECG1, ECG2` 文件顺序。`pair_embedding_swap` 使 GRU 接收当前拍 160 维 CNN 表征，而非仅三个 logits；其余因果约束不变：窗口止于 `R_i+200 ms`、每条记录清状态、不读目标标签/统计量/阈值或未来波形、R 峰和 RR。

若要评估“单导联训练、双导联冻结迁移”的受限门控版本，使用 `scripts/run_singlelead_multiview_gate_3gpu.sh`。门控只读取两个当前 logits 与因果 RR，在 DS1 六折中选取，并施加 DS1 S-F1 不低于固定算术平均的约束；不会用外部域指标选择。

## 目录

```text
beatgru/
  data.py       # DS1/DS2 manifest、因果窗口、因果 RR、单导联选择
  models.py     # BeatCNN 与 LogitGRU
  train.py      # CNN / GRU 训练与测试
  audit.py      # 自动因果性审计
  metrics.py    # N/S/V 指标
scripts/        # 构建、三卡训练、汇总、审计
results/        # 已完成实验的原始指标快照
```

## 严格 M0–M3 消融归档

`runs/m0_m3_three_seed_20260918_v2/` 已完成固定 DS1 训练、DS2 唯一评估、INCART-II/SVDB-ECG1 冻结推理和 3 个 seed 的因果审计。若需要从已有 checkpoint 重新补齐逐拍 provenance、M0 logits、记录级 paired bootstrap 和图 3–5，可运行：

```bash
python tools/finalize_strict_ablation.py \
  --root runs/m0_m3_three_seed_20260918_v2 \
  --manifest-dir /path/to/manifest/v4_strict_causal \
  --external-manifest /path/to/manifest/v4_strict_causal \
  --device cuda
```

归档中的 `MANIFEST.json` 为只读文件；由于这套历史运行目录建立时没有记录 git commit，manifest 同时保存了核心代码和数据 manifest 的 SHA256 指纹，并明确标注训练启动时间未被旧 launcher 捕获。

## R 峰检测前端评估

BEATGRU 的主指标是条件于 R 峰的在线分类指标。要评估 WFDB `gqrs/xqrs` 前端在相同 beat manifest 上的独立误差，运行：

```bash
python tools/evaluate_rpeak_detector.py \
  --manifest-dir /path/to/manifest/v4_strict_causal \
  --output-dir runs/rpeak_detector_test_mit \
  --datasets mit_bih_arrhythmia --split test --tolerance-ms 100
```

该工具不会把检测结果混入分类指标；它单独输出每条记录和数据集级 TP/FP/FN、PPV、Sensitivity、F1 及定位误差。

## 可复现性说明

所有随机源均使用显式 seed，并启用 deterministic cuDNN。GPU/CUDA 版本不同可能产生末位浮点差异。训练固定 15 epochs，避免在 DS2 测试集上进行 checkpoint 或阈值选择；DS2 仅在训练冻结后加载一次用于最终评估。

## 引用与数据来源

请引用 MIT-BIH Arrhythmia Database 与 PhysioNet；如使用本代码，请说明采用标准 MIT-BIH inter-patient DS1/DS2 协议、R 峰条件分类和 `R_i+200 ms` 因果决策延迟。
