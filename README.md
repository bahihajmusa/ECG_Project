# Edge-Ready ECG Classification using CRISP-DM

Deep-learning classification of heartbeats (**N / S / V / F**) on the **MIT-BIH Arrhythmia Database**, designed for the
**Analog Devices MAX78002** Edge-AI chip. The project covers all six CRISP-DM stages and is split by **patient record**
(no leakage). **Task 1** (problem, data, preparation) and **Task 2** (modeling, evaluation, deployment) are delivered together in this folder, and
everything can be reproduced from it.

> **Important:** this is a course project about the data-mining and Edge-AI process. It is **not** a medical device and makes
> **no claim of clinical validation**. No MAX78002 board was used, so **no hardware measurements** (latency, energy, RAM use) are reported;
> deployment feasibility is shown with the official software toolchain only.

## Contents
1. [Summary](#1-summary)
2. [Folder structure](#2-folder-structure)
3. [Quick start](#3-quick-start-the-order-of-the-steps)
4. [Setup and data download](#4-setup-and-data-download)
5. [Task 1: problem, data and preparation](#5-task-1-problem-data-and-preparation)
6. [Task 2: modeling](#6-task-2-modeling)
7. [Task 2: evaluation and final results](#7-task-2-evaluation)
8. [Task 2: deployment on MAX78002](#8-task-2-deployment-on-max78002)
9. [CRISP-DM stages: where each is documented](#9-crisp-dm-stages-where-each-is-documented)
10. [Project rules and how the code enforces them](#10-project-rules-and-how-the-code-enforces-them)
11. [Reproducibility details](#11-reproducibility-details)
12. [Known limitations](#12-known-limitations)
13. [Credits](#13-credits)

---------------------------------------------------------------------------------------------------

## 1. Summary

| Item | Choice |
|---|---|
| Task | 4-class heartbeat classification: Normal (N), Supraventricular ectopic (S), Ventricular ectopic (V), Fusion (F) |
| Data | MIT-BIH Arrhythmia Database, 44 records (DS1 + DS2), lead MLII, 360 Hz; MIT-BIH Noise Stress Test Database for noise tests |
| Input | 360 samples (1 s) around the R-peak, band-pass 0.5-40 Hz, per-beat min-max to [-1, 1] |
| Split | Standard inter-patient split: DS1 (22 records) = 18 train + 4 validation, DS2 (22 records) = final test |
| Final dataset | 100,667 beats: train 40,721, validation 10,266, test 49,680 |
| Main model | **EdgeECGNet**, a 1D CNN (6 fused Conv1d + BatchNorm + ReLU layers, max-pool in front of layers 2-6, one linear layer), **25,268 parameters** (about 25 KB at 8 bits) |
| Baselines | Logistic Regression, Random Forest (`baseline.py`) and a very small CNN `tiny_cnn` (2,900 parameters) |
| Imbalance | Class-weighted cross-entropy (weights from the training set only, softened with power 0.5); balanced sampler as alternative (never both) |
| Augmentation | Training beats only: gain, small shift, baseline wander, Gaussian noise |
| Quantization | 8-bit quantization-aware training: PyTorch emulation **and** the official ai8x-training / ai8x-synthesis flow |
| Noise test | Real NSTDB noise (`bw`, `em`, `ma`) at 12, 6 and 0 dB on DS2 |
| Deployment evidence | Official `ai8xize` synthesis for MAX78002: **PASSED**, 25,040 bytes weight memory (1.0 %), 1,420,688 operations per inference |
| Final result (DS2) | Macro-F1 0.4019 (INT8 emulation), 0.4145 (FP32), 0.4919 (deployed ai8x 8-bit model); the macro-F1 and minority-class targets are **not met** (see section 7) |

The model is a **deep neural network** trained end to end, as the project document requires for the AI programme. All layers are operations
that the MAX78002 CNN accelerator supports.

---------------------------------------------------------------------------------------------------

## 2. Folder structure

```
GroupXX_ECG_Project/
├── report.pdf                      the report (Task 1 and Task 2)
├── README.md                       this file
├── requirements.txt
├── run_task2.sh                    runs the whole Task 2 pipeline
├── report_notes.yaml               summary, success-criteria targets, run history, list of changes (used by make_report.py)
├── report_task1.md                 text of report sections 1-3 and the references (used by make_report.py)
├── report_assets/                  logo for the title page
├── notebooks/
│   └── EDA.ipynb                   exploratory data analysis (Task 1), saved with outputs
├── src/
│   ├── prepare_data.py             download, preprocessing, split, statistics tables           (Task 1)
│   ├── dataset.py                  PyTorch Dataset, training-only augmentation, sampler        (Task 1)
│   ├── utils.py                    shared helpers: metrics, plots, config
│   ├── model.py                    EdgeECGNet, MAX78002 checks, INT8 (PTQ / QAT) emulation     (Task 2)
│   ├── train.py                    FP32 training, QAT, model selection (sweep)
│   ├── baseline.py                 Logistic Regression / Random Forest baselines
│   ├── evaluate.py                 metrics, confusion matrices, tables 5.1-5.3
│   ├── noise_test.py               noise robustness (5.4) + band-pass filter ablation
│   ├── error_analysis.py           error analysis (5.5)
│   ├── export_ai8x.py              generates the MAX78002 (ai8x) files from the model definition
│   ├── export_eda_figures.py       extracts the figures saved in notebooks/EDA.ipynb into results/eda/
│   └── make_report.py              builds report/Full_report.pdf (Task 1 + Task 2 in one report) from the project files
├── configs/
│   ├── edge_cnn.yaml               starting configuration of the final model + training + QAT settings
│   ├── best_config.yaml            FROZEN final configuration (sweep winner + validation-only refinements)
│   ├── tiny_cnn.yaml               very small CNN baseline
│   └── sweep.yaml                  model-selection experiments (validation only)
├── report/                         generated: Full_report.pdf
├── results/                        generated: metrics.csv, confusion_matrix_*, noise_results.*, tables, figures
├── models/                         generated: trained models
├── data/                           generated: raw and processed data (NOT submitted)
└── max78002/
    ├── model_configuration/        ai8x model, dataset, network YAML, policies, scripts
    ├── quantization_output/        generated: 8-bit checkpoint and logs
    ├── synthesis_output/           generated: C project and synthesis_summary.md
    ├── run_toolchain.sh            runs the Analog Devices flow
    ├── collect_evidence.py         writes synthesis_summary.md
    └── README_toolchain.md         details of the MAX78002 flow
```
The ai8x repositories themselves (`ai8x-training`, `ai8x-synthesis`) are separate software and are **not** part of the submission.

---------------------------------------------------------------------------------------------------

## 3. Quick start: the order of the steps

```bash
pip install -r requirements.txt                              # 0. install once (Python 3.9 - 3.12)
cd src && python prepare_data.py && cd ..                    # 1. download + prepare the data (Task 1)
#    open notebooks/EDA.ipynb, run all cells, save with outputs (Task 1)
QUICK=1 sh run_task2.sh                                      # 2. smoke test: everything runs (a few minutes, results are NOT meaningful)
rm -rf results/* models/*                                    #    delete the smoke-test results before the real run
SWEEP=1 sh run_task2.sh                                      # 3. real run: baselines, model selection, final model (FP32 + QAT), validation results
ONLY_FINAL=1 sh run_task2.sh                                 # 4. ONCE, when you are happy: DS2 results, noise test, error analysis (no re-training)
AI8X_TRAINING=... AI8X_SYNTHESIS=... sh max78002/run_toolchain.sh   # 5. MAX78002 toolchain (own Python 3.11 environment)
cd src && python export_eda_figures.py                       # 6. figures from the EDA notebook (run the notebook first, save with outputs)
python make_report.py --config ../configs/best_config.yaml --split test   # 7. build report/Full_report.pdf (Task 1 + Task 2)
```
* The MAX78002 toolchain (step 5) is **not** needed for steps 1-4 and is run **after** the model is chosen (step 3), because it is generated from the final
  model definition. `FINAL=1` is added only at the very end of step 5 (see section 8).
* `run_task2.sh` uses `configs/best_config.yaml` automatically once it exists. Options: `SWEEP=1` (model selection), `ONLY_FINAL=1` (final
  steps without training), `FINAL=1` (training and final steps together), `QUICK=1` (very short training).
* Steps 6 and 7 only read files in `results/` and `max78002/`; it never trains or evaluates, so it cannot touch DS2 data. Edit `report_task1.md` (text of sections 1-3) and `report_notes.yaml` (summary, targets, versions) first; everything else in the PDF is generated.
* On Windows use Git Bash or WSL for the `.sh` scripts, or run the Python commands of sections 5 and 6 one by one. The MAX78002 toolchain is separate software
  and best run on Linux/macOS/WSL.

---------------------------------------------------------------------------------------------------

## 4. Setup and data download

```bash
python -m venv .venv && source .venv/bin/activate       # optional
pip install -r requirements.txt                          # includes reportlab for make_report.py
```

The datasets are **not** part of the submission. `prepare_data.py` downloads them with the `wfdb` package from PhysioNet:

```bash
cd src
python prepare_data.py --data-dir ../data/raw --output-dir ../data/processed
# optional: also write noisy copies of the test set (about 70 MB each, 9 sets)
python prepare_data.py --make-noisy-test --noise-dir ../data/nstdb --data-dir ../data/raw --output-dir ../data/processed
```
Manual download: MIT-BIH Arrhythmia Database <https://physionet.org/content/mitdb/1.0.0/> into `data/raw/`; MIT-BIH Noise Stress Test Database
<https://physionet.org/content/nstdb/1.0.0/> (records `bw`, `em`, `ma`) into `data/nstdb/`; then run `python prepare_data.py --skip-download`.
`noise_test.py` can also build the noisy test sets on the fly, so `--make-noisy-test` is optional.

---------------------------------------------------------------------------------------------------

## 5. Task 1: problem, data and preparation

Full text in the report (sections 1-3). Here is the summary and how to reproduce it.

### 5.1 Problem and success criteria
Classify each heartbeat as N, S, V or F from one second of ECG, for a wearable Edge-AI device (low latency, privacy, battery, offline use).
Success criteria (report 1.4), measured on DS2: overall accuracy and macro-F1, per-class recall/F1 for V, S and F, macro-F1 drop under 12/6/0 dB noise,
model size (at most 2 MB, goal about 100,000 parameters or fewer), FP32-to-INT8 macro-F1 drop, MAX78002 compatibility, and process quality (no record in two splits).
Accuracy alone is not used: an "always N" classifier already gets about 89 % accuracy on DS2.

### 5.2 Data understanding (`notebooks/EDA.ipynb`)
* 44 records (22 DS1 + 22 DS2, 43 subjects; records 201 and 202 are the same person), 2 channels, 360 Hz, about 30 minutes each, 103,724 expert annotations.
* **Lead:** MLII exists in all 44 records. It is channel 1 in record 114 only, so the code selects the lead **by name**.
* **Quality:** no missing values, no flat windows; record 116 touches the recorder limit (977 samples beyond 5 mV); 38 of 44 records contain expert noise marks (`~`, `|`);
  records 108 and 203 have the most drift; signal spread differs strongly between records (0.17-0.67 mV).
* **Imbalance:** N 89.5 %, S 2.8 %, V 7.0 %, F 0.8 %; 12 records have no S beat and 27 have no F beat.
* **Leakage analysis:** a random beat split gave macro-F1 0.87; splitting by record gave 0.33 (small demonstration with a Random Forest). The project therefore splits by record only.

### 5.3 Data preparation (`src/prepare_data.py`)
Pipeline: `raw MLII -> band-pass filter -> cut 360 samples around each R-peak -> drop flat beats -> per-beat min-max to [-1, 1]`.

| Step | Setting | Reason |
|---|---|---|
| Lead | MLII, selected by name (fallback to channel 0 is flagged) | same lead in all records |
| Filter | Butterworth band-pass 0.5-40 Hz, order 4, zero-phase (`--causal` for real-time) | removes baseline wander and muscle noise |
| Window | 180 samples before, R, 179 after = 360 samples (1 s) | covers P-QRS-T; fixed network input |
| Flat-beat check | peak-to-peak below 0.001 mV removed (none found) | no information |
| Normalisation | per beat, min-max to [-1, 1] | removes gain differences; fits signed 8-bit on the chip |
| Clipping | **not used** (`--clip-sigma` to enable) | a 5-sigma record clip would cut the R-peak of a median 26 % of beats |

**Labels (AAMI EC57):** N = N, L, R, e, j; S = A, a, J, S; V = V, E; F = F. Excluded and counted: unclassifiable `Q` (15) and non-beat marks `+ ~ ! " x | [ ]` (2,991).
51 beats were lost because their window leaves the recording. Every exclusion is in `label_mapping_counts.csv`.

**Split (by record):**

| Split | Records | N | S | V | F | Total |
|---|---|---|---|---|---|---|
| Train (18 records of DS1) | 101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 | 36,398 | 774 | 3,150 | 399 | 40,721 |
| Validation (4 records of DS1) | 215 220 223 230 | 9,443 | 170 | 638 | 15 | 10,266 |
| Test = DS2 (22 records) | 100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234 | 44,235 | 1,837 | 3,220 | 388 | 49,680 |
| **All** | 44 records | 90,076 | 2,781 | 7,008 | 802 | 100,667 |

**Class weights** (training set only): N 0.2797, S 13.1528, V 3.2318, F 25.5144. The final model uses them with power 0.5 (square root), see 6.3.
**Augmentation** (training beats only, `dataset.py`): amplitude gain 0.9-1.1, shift up to 3 samples (border fill), baseline wander (0.1-0.5 Hz, up to 10 % of the beat range), Gaussian noise (2 % of the beat std).
**Noise protocol:** NSTDB recordings are cut in time; the first half is for training augmentation, the second half only for the DS2 noise test. Noise is added to the raw ECG before filtering.

### 5.4 Files written by `prepare_data.py` (`data/processed/`)
`X_{train,val,test}.npy`, `y_*.npy`, `*_record_ids.npy`, `*_symbols.npy`, `*_r_peaks.npy`, `dataset_statistics.csv` (table 3.8), `per_record_statistics.csv`,
`label_mapping_counts.csv` (table 2.2), `class_weights.json`, `label_map.json`, `split_records.json`, `preprocessing_config.json` (every parameter and software version), and with `--make-noisy-test` also `X_test_<noise>_<snr>dB.npy`.

Options: `--skip-download`, `--seed N` (default 42), `--normalization {minmax,zscore,none}`, `--clip-sigma K`, `--causal`, `--make-noisy-test`, `--noise-types`, `--snr-db`.
Self-test of the dataset code: `python src/dataset.py` (prints "All tests passed!").

---------------------------------------------------------------------------------------------------

## 6. Task 2: modeling

### 6.1 Baselines (4.1)
* **Logistic Regression** and **Random Forest** (`baseline.py`): the 360 beat samples as features, class-balanced; the better one on the **validation** records is saved as `models/baseline_best.joblib`.
* **Tiny CNN** (`configs/tiny_cnn.yaml`): 3 conv layers, 2,900 parameters, same training recipe: a small deep-learning reference.

### 6.2 Proposed model: EdgeECGNet (4.2)
Only MAX78002-supported operations: 1D convolutions (kernel 1-5), max pooling, ReLU, BatchNorm (folded into the convolution for deployment) and one linear layer.
Max-pooling is applied **before** each convolution, as in the ai8x fused layers.

| Layer | Input (C x L) | Operation | Output | Parameters (deployed) |
|---|---|---|---|---|
| conv1 | 1 x 360 | Conv1d(k=5) + BN + ReLU | 16 x 360 | 96 |
| conv2 | 16 x 360 | MaxPool(2) -> Conv1d(k=5) + BN + ReLU | 32 x 180 | 2,592 |
| conv3 | 32 x 180 | MaxPool(2) -> Conv1d(k=5) + BN + ReLU | 32 x 90 | 5,152 |
| conv4 | 32 x 90 | MaxPool(2) -> Conv1d(k=3) + BN + ReLU | 48 x 45 | 4,656 |
| conv5 | 48 x 45 | MaxPool(2) -> Conv1d(k=3) + BN + ReLU | 64 x 22 | 9,280 |
| conv6 | 64 x 22 | MaxPool(2) -> Conv1d(k=1) + BN + ReLU | 32 x 11 | 2,080 |
| fc | 352 | Flatten -> Linear (32-bit output) | 4 | 1,412 |
| **Total** | | | | **25,268** (25,492 trainable with BatchNorm) |

Why: small (about 25 KB at 8 bits, far below 2 MB), supported layers only, fixed input size, full time resolution in the first layers for the sharp QRS shape.
`python src/model.py` prints this table; `check_max78002_constraints` tests kernel, padding, pooling, layer-count and memory rules.

### 6.3 Training (4.3, `configs/best_config.yaml`)
Final configuration (frozen after the validation-only runs of 6.4):

| Setting | Value |
|---|---|
| Optimizer / learning rate | Adam, 1e-3, cosine decay |
| Weight decay / dropout | 0.0 / 0.1 |
| Batch size / epochs | 128 / 30 |
| Loss | class-weighted cross-entropy, `class_weight_power: 0.5` |
| Gradient clipping | 1.0 |
| Model selection / early stopping | best **validation macro-F1** from epoch 1; patience 30, minimum 10 epochs |
| QAT | 30 epochs, learning rate x 0.1, patience 12, BatchNorm folded and frozen, 8-bit fake quantization with a straight-through estimator |
| Seed | 42 |

Augmentation on training beats only. `train.py` never loads the test set. The FP32 model had its best validation macro-F1 (0.4560) at epoch 5; the best QAT epoch was 0,
which means QAT did not improve on the converted FP32 model, so in this project `edge_int8_qat` and `edge_int8_ptq` give the same numbers.

### 6.4 Model selection (4.4, `configs/sweep.yaml`)
Seven short FP32 experiments: tiny CNN; EdgeECGNet width 0.5 / 1.0 / 1.5; learning rate 3e-3; balanced sampler instead of class weights; softer class weights (power 0.5).
**Rule:** highest **validation** macro-F1; if several are within 0.005, the one with the fewest parameters. The test set is not used. All runs are in `results/model_selection.csv`.
The sweep chose width 1.0 with softer class weights (validation macro-F1 0.514). After the sweep, a few further **validation-only** runs (dropout, weight decay, number of epochs;
run D = dropout 0.1, weight decay 0) fixed `configs/best_config.yaml`; one stronger-regularisation run (dropout 0.3, weight decay 1e-4) gave no gain and was rejected.
The validation set has only 15 F beats, so F-class differences are noisy; say so when you discuss the choice.
After the single DS2 evaluation, **no retuning was done**, because that would use DS2 for selection.

### 6.5 Quantization (4.5)
1. **PyTorch emulation** (`model.py`): BatchNorm folded, 8-bit weights/biases with a power-of-two scale per layer, 8-bit activations in [-1, 127/128], wide 32-bit output.
   `edge_int8_ptq` = post-training quantization of the FP32 model; `edge_int8_qat` = quantization-aware training. **Emulation only, not bit-exact with the chip.**
2. **Official toolchain** (`max78002/`): ai8x-training QAT (trained from scratch with the same class weights) -> `quantize.py` -> 8-bit evaluation -> `ai8xize.py`.
   Official 8-bit numbers: `max78002/quantization_output/` (see section 8).

### 6.6 Run Task 2 step by step (from `src/`)
```bash
python baseline.py   --config ../configs/edge_cnn.yaml                                  # LR + Random Forest, validation only
python train.py      --config ../configs/tiny_cnn.yaml                                  # tiny CNN baseline
python train.py      --config ../configs/edge_cnn.yaml --sweep ../configs/sweep.yaml    # model selection -> configs/best_config.yaml
python train.py      --config ../configs/best_config.yaml --name edge_cnn               # FP32, then QAT
python evaluate.py   --config ../configs/best_config.yaml --split val                   # validation tables
# ---- only when every decision is final ----
python evaluate.py   --config ../configs/best_config.yaml --split test --final          # DS2 clean results (5.1-5.3)
python noise_test.py --config ../configs/best_config.yaml --final --ablate-filter       # noise robustness (5.4)
python error_analysis.py --config ../configs/best_config.yaml --final                   # error analysis (5.5)
python make_report.py    --config ../configs/best_config.yaml --split test              # PDF report
```
(Without the sweep use `../configs/edge_cnn.yaml`; the shell script `run_task2.sh` does this choice for you. The sweep overwrites `best_config.yaml`; keep a copy of the frozen file.)
Override single values with `--set training.lr=0.003 training.epochs=40`. Use `device: cpu` in the config if a GPU / Apple MPS device gives problems.
The final run took about 9 minutes (520 s) on a CPU.

---------------------------------------------------------------------------------------------------

## 7. Task 2: evaluation

* `evaluate.py`: tables 5.1 (accuracy, macro-F1, parameters for baseline / FP32 / INT8), 5.2 (per-class precision, recall, F1) and 5.3 (FP32 versus INT8 and agreement of their predictions). Accuracy is never reported alone.
* `noise_test.py`: clean, 12, 6 and 0 dB for `bw`, `em` and `ma` (mean and per type), per-class F1 versus noise, and with `--ablate-filter` whether the band-pass filter helps at test time.
  Noisy beats are the same beats in the same order as the clean test set.
* `error_analysis.py`: main confusions, error rate per original MIT-BIH symbol and per record, example figures with ECG context, effect of noise.
* `make_report.py`: builds `report/Full_report.pdf` including the success-criteria review (5.6) from `results/`, the ai8x evaluation logs and the targets in `report_notes.yaml`.

| File in `results/` | Content (report section) |
|---|---|
| `metrics.csv`, `tables_val.md`, `tables_test.md` | accuracy, macro-F1, per-class metrics, ready-made tables (5.1-5.3) |
| `confusion_matrix_<model>_<split>.png/.csv` | confusion matrices |
| `model_selection.csv` | sweep experiments (4.4) |
| `training_log_*.csv`, `training_curves_*.png`, `training_summary_*.json` | training history (4.3) |
| `noise_results.csv`, `noise_results_table.md`, `noise_robustness.png`, `noise_per_class_f1.png`, `noise_filter_ablation.png` | noise robustness (5.4) |
| `error_analysis.md`, `error_examples_*.png`, `error_mean_beats.png`, `error_noise_examples.png` | error analysis (5.5) |
| `test_set_access_log.txt` | every use of the final test set DS2 |

### Final results on DS2 (22 test records, one evaluation)

| | Baseline (best classical) | Tiny CNN (FP32) | EdgeECGNet FP32 | INT8 (QAT = PTQ, emulated) |
|---|---|---|---|---|
| Accuracy | 0.8956 | 0.9204 | 0.9035 | 0.8967 |
| Macro-F1 | 0.4012 | 0.4970 | 0.4145 | 0.4019 |
| F1 N / S / V / F | 0.943 / 0.011 / 0.649 / 0.002 | 0.961 / 0.109 / 0.832 / 0.086 | 0.958 / 0.067 / 0.621 / 0.012 | 0.955 / 0.066 / 0.587 / 0.000 |
| Macro-F1 at 12 / 6 / 0 dB (mean of bw, em, ma) | 0.398 / 0.393 / 0.385 | 0.487 / 0.471 / 0.444 | 0.406 / 0.399 / 0.388 | 0.395 / 0.390 / 0.380 |
| Parameters | 208,796 | 2,900 | 25,268 | 25,268 |

Official ai8x 8-bit model (trained separately with ai8x-training, evaluated once on DS2): accuracy 0.8758, macro-F1 0.4919, F1 N / S / V / F 0.931 / 0.179 / 0.856 / 0.003, recall S 0.196, V 0.873, F 0.008.
On the validation records the same model reached accuracy 0.9316 and macro-F1 0.4933. It is a different model from the PyTorch INT8 emulation (own training run), so the two are reported side by side, not as one result.

**The two INT8 models: how they compare (DS2)**

| | PyTorch INT8 emulation (main model) | ai8x 8-bit (deployed model) |
|---|---|---|
| Accuracy / macro-F1 | 0.8967 / 0.4019 | 0.8758 / **0.4919** |
| F1 N / S / V / F | 0.955 / 0.066 / 0.587 / 0.000 | 0.931 / **0.179** / **0.856** / 0.003 |
| Recall S / V / F | 0.044 / 0.767 / 0.000 | **0.196** / **0.873** / 0.008 |

* The ai8x model is the one that was synthesized for the MAX78002. It is a separate training run (from scratch, checkpoint chosen on validation accuracy), **not** a more exact copy of the emulated model.
* It finds more V and S beats and pays with more false alarms (1,635 N beats labelled F, 1,815 labelled S); F is still almost never found. It also scored higher on validation (0.4933 versus 0.4551), so the gain is not only a DS2 effect.
* Each model comes from one run, so part of the difference may be run-to-run variation. The emulation stays the main model because it was planned before DS2 was opened; the ai8x result was not used to change any decision.
* The ai8x model scores about the same as the 2,900-parameter tiny CNN, so training recipe and model selection matter more than network size.

**Reading the numbers honestly**
* The macro-F1 target and the minority-class targets (S, F) are **not met**. The edge model is only marginally better than the classical baseline in macro-F1, and the tiny CNN is better than the edge model on DS2.
  The larger model is therefore justified by the INT8 and MAX78002 analysis, not by accuracy.
* FP32 to INT8 loses 0.0126 macro-F1; noise lowers macro-F1 gradually (electrode-motion noise hurts most); the band-pass filter helps (0.402 with, 0.361 without on clean data).
* Errors are concentrated: records 232 and 213 hold most S and F failures, and bundle-branch-block beats (L, R) are often predicted as V. One 1-second window carries no RR-interval (timing) information, which S and F beats depend on.
* Model size (25 KB INT8), FP32-to-INT8 gap, MAX78002 compatibility and process quality (no record in two splits) are met.

---------------------------------------------------------------------------------------------------

## 8. Task 2: deployment on MAX78002

Details in `max78002/README_toolchain.md`. Set up `ai8x-training` and `ai8x-synthesis` once (separate software, **own Python 3.11 environment**, CPU-only PyTorch is enough), then from the project root:

```bash
AI8X_TRAINING=/path/to/ai8x-training AI8X_SYNTHESIS=/path/to/ai8x-synthesis sh max78002/run_toolchain.sh
# FINAL=1 additionally evaluates the 8-bit model on DS2 (once, at the very end). It re-runs the whole flow including training.
# In this project the DS2 evaluation was run on the existing quantized model instead (ECG_TEST_SET=ds2 sh model_configuration/scripts/evaluate_ecg.sh), see below.
```
Before the first run generate the ai8x files from the frozen configuration (in the project environment):
```bash
cd src && python export_ai8x.py --config ../configs/best_config.yaml --out ../max78002 --epochs 40 --qat-start 25
```
`export_ai8x.py` applies `class_weight_power` to the class weights (here 0.5289 / 3.6267 / 1.7977 / 5.0512). The script `run_toolchain.sh` then copies the files into the two repositories, trains with QAT, quantizes, evaluates the 8-bit model and synthesizes the C project;
`collect_evidence.py` writes `max78002/synthesis_output/synthesis_summary.md` (PASSED/FAILED, weight memory, bias memory, operations per layer).

* **6.1 Compatibility:** layers, kernels, padding, pooling, channels and the 2 MB weight limit are checked in `model.py` and finally by the synthesis tool.
* **6.2 Toolchain:** ai8x-training (QAT) -> `quantize.py` -> `ai8xize.py` -> generated C code.
* **6.3 Evidence (own run):** **Synthesis PASSED.** Weight memory 25,040 of 2,396,160 bytes (1.0 %), bias memory 228 of 8,192 bytes (2.8 %), 1,420,688 operations per inference (1,384,448 multiply-accumulates),
  estimated peak activation memory 11,520 bytes (an estimate). The generated C project (`cnn.c`, `cnn.h`, `main.c`, `weights.h`, `sampledata.h`, `sampleoutput.h`) is in `synthesis_output/`.
  The 8-bit ai8x evaluation (best epoch 35) gave accuracy 93.16 % and macro-F1 0.4933 on validation, and 87.58 % and 0.4919 on DS2 (one evaluation); both logs are in `quantization_output/` (`ai8x_evaluation_validation.log`, `ai8x_evaluation_test_DS2.log`).
* **6.4 Hardware limitations:** no board was used; latency, energy, real RAM use and the input pipeline (ADC, R-peak detection, on-device filtering) are **not measured**.
  The operation count is a simulation-level quantity only. The zero-phase filter uses future samples; a real device needs a causal filter (`prepare_data.py --causal`).
* **6.5 Proposed hardware validation (not done):** flash the generated project to a MAX78002 evaluation kit, run the DS2 test beats through the chip and compare with the INT8 reference,
  measure latency and energy per inference, and test with ECG from a real wearable front end.

---------------------------------------------------------------------------------------------------

## 9. CRISP-DM stages: where each is documented

| Stage | Where |
|---|---|
| 1. Business / application understanding | report section 1; section 5.1 of this README |
| 2. Data understanding | `notebooks/EDA.ipynb`, report section 2; section 5.2 |
| 3. Data preparation | `src/prepare_data.py`, `src/dataset.py`, `data/processed/*.csv/json`, report section 3; section 5.3 |
| 4. Modeling | `src/model.py`, `train.py`, `baseline.py`, `configs/`, `results/model_selection.csv`, report section 4; section 6 |
| 5. Evaluation | `evaluate.py`, `noise_test.py`, `error_analysis.py`, `results/`, report section 5; section 7 |
| 6. Deployment | `max78002/`, report section 6; section 8 |

---------------------------------------------------------------------------------------------------

## 10. Project rules and how the code enforces them

| Rule | How |
|---|---|
| No beats of one recording in both train and test | Split by record; assertions on the real beat arrays; per-beat record ids saved |
| DS2 not used for selection or tuning | `train.py` / `baseline.py` never load it; the ai8x DS2 files are copied only for the one final evaluation and then removed; `evaluate.py --split test`, `noise_test.py`, `error_analysis.py` refuse to run without `--final`; each use is logged in `results/test_set_access_log.txt`; no retuning after seeing DS2 results |
| No hyperparameter search on test results | The sweep and all refinement runs use validation macro-F1 only |
| Not accuracy alone | Macro-F1, per-class precision/recall/F1 and confusion matrices everywhere |
| No leakage through statistics or noise | Per-beat normalisation, class weights from the training set only, noise split in time |
| No clinical claims, no hardware measurements claimed | Stated at the top of this README and in the report |
| MAX78002-compatible model | Layer set, `check_max78002_constraints`, real `ai8xize` synthesis (PASSED) |
| Reproducible preprocessing, exclusions, class mapping | `prepare_data.py`, `label_mapping_counts.csv`, `preprocessing_config.json`, seed 42 |

---------------------------------------------------------------------------------------------------

## 11. Reproducibility details
* Seed 42 (Python, numpy, torch; each data-loader worker gets its own seed). Settings are in `configs/*.yaml` and `data/processed/preprocessing_config.json`.
* `preprocessing_config.json` records the Python, numpy, scipy, pandas and wfdb versions of the preparation run. Run `prepare_data.py`, the notebook and the training
  in **the same Python environment**, so all versions agree; write the versions into the report.
* The ai8x toolchain ran in a separate environment (Python 3.11, torch 2.3.1 CPU, numpy 1.26.4); `ai8xize.py` is called with `--no-version-check`.
* Results of single runs can differ slightly between machines (different CPU/GPU libraries), even with a fixed seed.
* Exact commands for every result are in sections 4-8.

## 12. Known limitations
* Records 201 (train) and 202 (test) are the same person; the standard split keeps this.
* The validation set (4 records) has very few F beats (15), so validation metrics for F are noisy.
* S and F beats are largely missed on DS2 (recall 4 % for S, 0 % for F in the emulated INT8 model; 20 % and 1 % in the ai8x model); the input has no RR-interval information.
* Each INT8 model comes from a single training run (one seed), so the difference between the emulated and the ai8x model may partly be run-to-run variation.
* R-peak positions come from the expert annotations, not from an automatic detector.
* Beats near expert-marked noisy regions are kept in the data.
* Historical dataset, no prospective clinical test, no hardware test, possible domain shift to wearable ECG devices.
* The software INT8 emulation is not bit-exact with the chip; the official toolchain numbers are the reference. QAT gave no gain over post-training quantization here.
* Zero-phase filtering is not possible in real time.

## 13. Credits
MIT-BIH Arrhythmia Database and MIT-BIH Noise Stress Test Database (PhysioNet; Goldberger et al., 2000; Moody and Mark, 2001; Moody et al., 1984).
WFDB Python package. Analog Devices `ai8x-training` and `ai8x-synthesis`. ReportLab for the PDF report.
