# Noise robustness (DS2 test set)

## baseline: mean over the noise types (bw, em, ma)

| Condition | Accuracy | Macro F1 | F1 N | F1 S | F1 V | F1 F |
|---|---|---|---|---|---|---|
| Clean | 0.8956 | 0.4012 | 0.943 | 0.011 | 0.649 | 0.002 |
| 12 dB | 0.8928 | 0.3978 | 0.941 | 0.009 | 0.640 | 0.002 |
| 6 dB | 0.8903 | 0.3933 | 0.940 | 0.009 | 0.623 | 0.002 |
| 0 dB | 0.8862 | 0.3852 | 0.937 | 0.005 | 0.596 | 0.003 |

## tiny_cnn_fp32: mean over the noise types (bw, em, ma)

| Condition | Accuracy | Macro F1 | F1 N | F1 S | F1 V | F1 F |
|---|---|---|---|---|---|---|
| Clean | 0.9204 | 0.4970 | 0.961 | 0.109 | 0.832 | 0.086 |
| 12 dB | 0.9105 | 0.4868 | 0.955 | 0.101 | 0.815 | 0.076 |
| 6 dB | 0.8951 | 0.4711 | 0.946 | 0.093 | 0.783 | 0.063 |
| 0 dB | 0.8717 | 0.4439 | 0.932 | 0.083 | 0.713 | 0.048 |

## edge_fp32: mean over the noise types (bw, em, ma)

| Condition | Accuracy | Macro F1 | F1 N | F1 S | F1 V | F1 F |
|---|---|---|---|---|---|---|
| Clean | 0.9035 | 0.4145 | 0.958 | 0.067 | 0.621 | 0.012 |
| 12 dB | 0.8938 | 0.4057 | 0.952 | 0.062 | 0.596 | 0.012 |
| 6 dB | 0.8831 | 0.3988 | 0.946 | 0.060 | 0.578 | 0.011 |
| 0 dB | 0.8641 | 0.3879 | 0.933 | 0.056 | 0.547 | 0.015 |

## edge_int8_ptq: mean over the noise types (bw, em, ma)

| Condition | Accuracy | Macro F1 | F1 N | F1 S | F1 V | F1 F |
|---|---|---|---|---|---|---|
| Clean | 0.8967 | 0.4019 | 0.955 | 0.066 | 0.587 | 0.000 |
| 12 dB | 0.8879 | 0.3954 | 0.950 | 0.063 | 0.566 | 0.002 |
| 6 dB | 0.8773 | 0.3895 | 0.944 | 0.060 | 0.550 | 0.004 |
| 0 dB | 0.8584 | 0.3799 | 0.931 | 0.056 | 0.524 | 0.009 |

## edge_int8_qat: mean over the noise types (bw, em, ma)

| Condition | Accuracy | Macro F1 | F1 N | F1 S | F1 V | F1 F |
|---|---|---|---|---|---|---|
| Clean | 0.8967 | 0.4019 | 0.955 | 0.066 | 0.587 | 0.000 |
| 12 dB | 0.8879 | 0.3954 | 0.950 | 0.063 | 0.566 | 0.002 |
| 6 dB | 0.8773 | 0.3895 | 0.944 | 0.060 | 0.550 | 0.004 |
| 0 dB | 0.8584 | 0.3799 | 0.931 | 0.056 | 0.524 | 0.009 |

## edge_int8_qat: per noise type

| Noise type | SNR | Accuracy | Macro F1 |
|---|---|---|---|
| baseline wander (bw) | 12 dB | 0.8953 | 0.4007 |
| baseline wander (bw) | 6 dB | 0.8948 | 0.3997 |
| baseline wander (bw) | 0 dB | 0.8923 | 0.3980 |
| electrode motion (em) | 12 dB | 0.8779 | 0.3898 |
| electrode motion (em) | 6 dB | 0.8480 | 0.3756 |
| electrode motion (em) | 0 dB | 0.7955 | 0.3530 |
| muscle artifact (ma) | 12 dB | 0.8906 | 0.3959 |
| muscle artifact (ma) | 6 dB | 0.8890 | 0.3932 |
| muscle artifact (ma) | 0 dB | 0.8874 | 0.3887 |

## Band-pass filter ablation (main model, mean macro-F1)

| Condition | filter ON | filter OFF |
|---|---|---|
| Clean | 0.4019 | 0.3614 |
| 12 dB | 0.3954 | 0.3632 |
| 6 dB | 0.3895 | 0.3635 |
| 0 dB | 0.3799 | 0.3575 |