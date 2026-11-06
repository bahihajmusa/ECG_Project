# Error analysis (edge_int8_qat, DS2 test set)

Beats: 49680 | accuracy 0.8967 | macro-F1 0.4019 | errors: 5131

## 1. Most frequent confusions

| True | Predicted | Beats | % of the true class |
|---|---|---|---|
| N | V | 1703 | 3.8% |
| S | V | 1017 | 55.4% |
| S | N | 716 | 39.0% |
| V | N | 644 | 20.0% |
| N | S | 437 | 1.0% |
| F | N | 368 | 94.8% |
| N | F | 97 | 0.2% |
| V | S | 64 | 2.0% |

* Class S: 4.4% recall; 58% of its errors are predicted as V.
* Class V: 76.7% recall; 86% of its errors are predicted as N.
* Class F: 0.0% recall; 95% of its errors are predicted as N.

## 2. Error rate per original MIT-BIH symbol

| Symbol | Class | Beats | Errors | Error rate | Most common wrong class |
|---|---|---|---|---|---|
| N | N | 36423 | 720 | 2.0% | S |
| L | N | 4124 | 1057 | 25.6% | V |
| R | N | 3475 | 448 | 12.9% | V |
| V | V | 3219 | 748 | 23.2% | N |
| A | S | 1736 | 1673 | 96.4% | V |
| F | F | 388 | 388 | 100.0% | N |
| j | N | 213 | 12 | 5.6% | V |
| J | S | 51 | 51 | 100.0% | N |
| a | S | 50 | 33 | 66.0% | N |
| E | V | 1 | 1 | 100.0% | N |

## 3. Records with the highest error rate

| Record | Beats | Errors | Error rate | Class counts N/S/V/F |
|---|---|---|---|---|
| 232 | 1780 | 1601 | 89.9% | 398/1382/0/0 |
| 213 | 3249 | 644 | 19.8% | 2639/28/220/362 |
| 111 | 2124 | 576 | 27.1% | 2123/0/1/0 |
| 214 | 2258 | 574 | 25.4% | 2001/0/256/1 |
| 222 | 2481 | 443 | 17.9% | 2272/209/0/0 |
| 228 | 2052 | 331 | 16.1% | 1687/3/362/0 |
| 212 | 2747 | 227 | 8.3% | 2747/0/0/0 |
| 233 | 3077 | 184 | 6.0% | 2229/7/830/11 |

The 3 records with most errors hold 55% of all errors.

## 4. Example figures

* `error_examples_N_to_S.png`: True N predicted as S (false alarms for S) (437 beats in total)
* `error_examples_S_to_N.png`: True S predicted as N (missed S) (716 beats in total)
* `error_examples_V_missed.png`: True V that the model missed (749 beats in total)
* `error_examples_F_missed.png`: True F that the model missed (388 beats in total)

* `error_mean_beats.png`: average beat of each class, correctly classified versus misclassified

## 5. Effect of noise (ma, 0 dB SNR)

Clean: accuracy 0.8967, macro-F1 0.4019. Noisy: accuracy 0.8874, macro-F1 0.3887.

* 1738 beats were correct when clean and wrong with noise; 1277 went the other way.
* Where the noise pushes the predictions: V (57%), S (21%), N (17%), F (4%)

| Class | Recall clean | Recall noisy | F1 clean | F1 noisy |
|---|---|---|---|---|
| N | 0.949 | 0.944 | 0.955 | 0.948 |
| S | 0.044 | 0.040 | 0.066 | 0.059 |
| V | 0.767 | 0.696 | 0.587 | 0.548 |
| F | 0.000 | 0.000 | 0.000 | 0.000 |

* `error_noise_examples.png`: beats that are correct when clean but wrong with noise

## 6. Points to discuss in the report

* Which class is hardest and why (small number of training beats, similar shape to N, patient differences)?
* Are the errors concentrated in a few records (section 3)?
* Do errors look like real ambiguity, label noise, or signal artifacts (see the example figures)?
* What does noise change: which classes lose most, and where do the wrong predictions go?