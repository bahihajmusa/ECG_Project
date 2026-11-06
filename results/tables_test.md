# Evaluation tables (test set)

## 5.1 Overall results

| Metric | Baseline (best classical) | Tiny CNN (FP32) | Edge model FP32 | INT8 post-training | INT8 QAT (emulated) |
|---|---|---|---|---|---|
| Accuracy | 0.8956 | 0.9204 | 0.9035 | 0.8967 | 0.8967 |
| Macro F1 | 0.4012 | 0.4970 | 0.4145 | 0.4019 | 0.4019 |
| Model parameters | 208,796 | 2,900 | 25,268 | 25,268 | 25,268 |

Baseline parameter count: Logistic Regression = weights + biases; Random Forest = total tree nodes.

## 5.2 Per-class performance

### Baseline (best classical)

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9347 | 0.9511 | 0.9429 | 44235 |
| S | 0.7692 | 0.0054 | 0.0108 | 1837 |
| V | 0.5728 | 0.7478 | 0.6487 | 3220 |
| F | 0.0022 | 0.0026 | 0.0024 | 388 |

### Tiny CNN (FP32)

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9483 | 0.9737 | 0.9608 | 44235 |
| S | 0.1524 | 0.0844 | 0.1086 | 1837 |
| V | 0.9154 | 0.7630 | 0.8323 | 3220 |
| F | 0.0731 | 0.1057 | 0.0864 | 388 |

### Edge model FP32

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9568 | 0.9583 | 0.9576 | 44235 |
| S | 0.1385 | 0.0441 | 0.0669 | 1837 |
| V | 0.5312 | 0.7475 | 0.6211 | 3220 |
| F | 0.0156 | 0.0103 | 0.0124 | 388 |

### INT8 post-training

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9605 | 0.9494 | 0.9549 | 44235 |
| S | 0.1363 | 0.0435 | 0.0660 | 1837 |
| V | 0.4747 | 0.7674 | 0.5866 | 3220 |
| F | 0.0000 | 0.0000 | 0.0000 | 388 |

### INT8 QAT (emulated)

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9605 | 0.9494 | 0.9549 | 44235 |
| S | 0.1363 | 0.0435 | 0.0660 | 1837 |
| V | 0.4747 | 0.7674 | 0.5866 | 3220 |
| F | 0.0000 | 0.0000 | 0.0000 | 388 |

## 5.3 Quantization analysis

* FP32 macro F1: 0.4145 | INT8 post-training (PTQ) macro F1: 0.4019 | difference: -0.0126
* FP32 macro F1: 0.4145 | INT8 QAT (emulated) macro F1: 0.4019 | difference: -0.0126
