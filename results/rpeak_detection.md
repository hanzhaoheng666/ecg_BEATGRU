# R 峰检测前端独立评估

评估脚本：`tools/evaluate_rpeak_detector.py`。参考峰来自 `v4_strict_causal` manifest 的时间顺序标注；预测使用 `hytrans.datasets.rpeak_detection.detect_rpeaks`，即 WFDB gqrs/xqrs，失败时回退到 scipy detector。匹配容差为 100 ms，一对一匹配。该结果不使用 beat 分类标签，也不混入 BEATGRU 的分类指标。

| 数据集 | test 记录数 | Reference peaks | Detected peaks | PPV | Sensitivity | F1 |
|---|---:|---:|---:|---:|---:|---:|
| MIT-BIH | 22 | 49,712 | 50,038 | 0.9925 | 0.9990 | **0.9957** |
| SVDB | 5 | 12,008 | 12,359 | 0.9694 | 0.9978 | **0.9834** |
| INCART | 5 | 11,321 | 11,313 | 0.9988 | 0.9981 | **0.9984** |
| MBIT | 3,161 | 30,111 | 43,330 | 0.2850 | 0.4101 | **0.3363** |

## 解释

- BEATGRU 的 DS2 结果仍然是“条件于已给定/检测到 R 峰”的 beat 分类结果。
- MIT-BIH、SVDB、INCART 可以把当前 detector 作为第一版在线前端，再单独报告检测误差。
- MBIT 的页面标签、采样协议和多导联信号与 MIT-BIH 差异明显；当前 detector 在 MBIT 上严重过检/漏检，不能直接把 MBIT 的分类结果解释成端到端结果。
- 后续 MBIT 主线应先固定一个不看测试标签的导联/预处理规则，再训练或校准专门的 R 峰 detector，并重新评估“检测峰 → BEATGRU 分类”的串联性能。

原始 JSON：`runs/rpeak_detector_test_full/rpeak_summary.json`，逐记录结果：`runs/rpeak_detector_test_full/rpeak_per_record.csv`。
