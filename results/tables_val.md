# Evaluation tables (val set)

## 5.1 Overall results

| Metric | Baseline (best classical) | Tiny CNN (FP32) | Edge model FP32 | INT8 post-training | INT8 QAT (emulated) |
|---|---|---|---|---|---|
| Accuracy | 0.9410 | 0.8730 | 0.8930 | 0.8780 | 0.8780 |
| Macro F1 | 0.4140 | 0.4054 | 0.4560 | 0.4551 | 0.4551 |
| Model parameters | 208,796 | 2,900 | 25,268 | 25,268 | 25,268 |

Baseline parameter count: Logistic Regression = weights + biases; Random Forest = total tree nodes.

## 5.2 Per-class performance

### Baseline (best classical)

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9405 | 0.9994 | 0.9690 | 9443 |
| S | 0.9091 | 0.1176 | 0.2083 | 170 |
| V | 0.9667 | 0.3182 | 0.4788 | 638 |
| F | 0.0000 | 0.0000 | 0.0000 | 15 |

### Tiny CNN (FP32)

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9541 | 0.9173 | 0.9353 | 9443 |
| S | 0.0733 | 0.3882 | 0.1234 | 170 |
| V | 0.9588 | 0.3652 | 0.5289 | 638 |
| F | 0.0227 | 0.0667 | 0.0339 | 15 |

### Edge model FP32

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9565 | 0.9304 | 0.9433 | 9443 |
| S | 0.0705 | 0.2235 | 0.1072 | 170 |
| V | 0.6393 | 0.5361 | 0.5831 | 638 |
| F | 0.3333 | 0.1333 | 0.1905 | 15 |

### INT8 post-training

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9581 | 0.9113 | 0.9341 | 9443 |
| S | 0.0616 | 0.2353 | 0.0977 | 170 |
| V | 0.5807 | 0.5752 | 0.5780 | 638 |
| F | 0.5000 | 0.1333 | 0.2105 | 15 |

### INT8 QAT (emulated)

| Class | Precision | Sensitivity/Recall | F1 | Support |
|---|---|---|---|---|
| N | 0.9581 | 0.9113 | 0.9341 | 9443 |
| S | 0.0616 | 0.2353 | 0.0977 | 170 |
| V | 0.5807 | 0.5752 | 0.5780 | 638 |
| F | 0.5000 | 0.1333 | 0.2105 | 15 |

## 5.3 Quantization analysis

* FP32 macro F1: 0.4560 | INT8 post-training (PTQ) macro F1: 0.4551 | difference: -0.0009
* FP32 macro F1: 0.4560 | INT8 QAT (emulated) macro F1: 0.4551 | difference: -0.0009
