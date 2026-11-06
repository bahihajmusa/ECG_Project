<!-- Text of Task 1 (sections 1-3) and the references; edited by hand. Task 2 is inserted by make_report.py at the marker line below. Double curly braces are replaced by values from report_notes.yaml. Figures: exclamation mark, caption in square brackets, file name and width in round brackets. -->
<!-- pagebreak -->
# 1. Business/Application Understanding
## 1.1 Application scenario
### What problem are you trying to solve?
The phrasing of the problem can be made by asking the question: “How do we use machine learning to detect abnormal heartbeats and classify the signals, all while still being able to be compatible with an Edge-AI device with a maximum size of 2MB?”.
### Where could automatic ECG classification be useful?
Because the primary deep learning algorithms (specifically CNNs in our case) can understand short-term patterns within individual heartbeats, and long-term irregularities spanning across multiple heartbeats, these algorithms can detect conditions like premature ventricular contraction and atrial fibrillation (Ansari et al., 2023). Automatic ECG would therefore, of course, be useful in healthcare. While automatic ECG could be used for both active testing and for monitoring in a clinical setting, it could also be implemented in sensors of mobile devices and smartwatches to detect abnormal signals even outside of clinical settings, alerting individuals to seek treatment. Because a 1D CNN requires relatively low computational power, it could work as lightweight monitoring built into smart devices.
### Why might inference directly on an Edge-AI device be useful?
Offline functionality, because the AI model runs locally there is no reliance on an internet-connection. This is also an important aspect in scenarios of crisis.
Then there is the aspect of data privacy. Because the data does not leave the device, it enforces the idea of privacy in the users data. Not to mention that it also makes it easier complying with data privacy laws.
Another aspect could be battery and data usage, sending a continuous ECG stream uses a lot of energy and bandwidth. A small chip like the MAX78002 can run a small neural network with very little energy.
### Who could potentially use such a system?
The potential users and key stakeholders for this system includes a broad number of individuals and medical professionals. Primarily, high-risk patients and elderly populations represent a core user group. Ramkumar et al. (2018) point out that atrial fibrillation prevalence exceeds 15% in individuals over 85 years old and is strongly linked to underlying conditions such as hypertension, diabetes, and prior stroke. These individuals could benefit from accessible, low-burden continuous screening during their daily routines (Niu et al., 2023). Furthermore, clinicians and healthcare providers serve as important users, as automated diagnostic triage tools can reduce clinical diagnostic workloads, standardize signal analysis, and assist cardiologists and primary care physicians in capturing sporadic, brief arrhythmic episodes that might otherwise go undetected during standard clinical visits (Ansari et al., 2023). The system also caters to the general public and health conscious consumers and as noted by Niu et al. (2023), everyday users of commercial smartwatches and smart wearables can engage in proactive personal health tracking, enabling the early detection of hidden or asymptomatic cardiac abnormalities.
## 1.2 Data-Mining Objective
### Input:
The input of the machine-learning model is an ECG heartbeat segment extracted from the MIT-BIH Arrhythmia Database. The segment is centered around an annotated heartbeat and consists of 360 samples (one second at 360 Hz: 180 before the R-peak, the R-peak itself and 179 after) of lead MLII, band-pass filtered at 0.5–40 Hz and normalised per beat to the range [-1, 1].
### Output:
The model predicts the class of the input heartbeat. Each heartbeat is assigned to one of four classes: N (Normal-type), S (Supraventricular ectopic), V (Ventricular ectopic), or F (Fusion).
### Type of task:
The problem is formulated as a supervised multiclass classification task, since the model is trained using ECG heartbeat samples with known class labels and must assign each previously unseen heartbeat to one of four possible classes.
## 1.3 Project Objectives
- Training of a model with a four-class heartbeat classification – N, S, V, & F.
- Create a model that will be compatible with MAX78002.
- Generalise to new ECG-recordings of people the model has never seen (test on DS2).
- Handle class imbalance, so rare classes are not ignored.
- Stay robust to ECG noise.
- Keep every step reproducible and documented.
## 1.4 Success Criteria
Accuracy alone is not a good measure here – if a model always answers “N”, it is right for 89% of the dataset test beats, but it will not find abnormal beats at all. So we use per class measures. The criteria below are measured on the final test set DS2. The numbers were formalised during Task 2 (the first submission described the criteria only in words); they are based on reference points (the “always N” classifier, the classical baseline and the 2 MB limit of the MAX78002) and were not used to make any modelling decision.
| Area | Measure on DS2 | Target |
|---|---|---|
| Predictive performance | Macro-F1 (always reported together with accuracy and the confusion matrix) | Macro-F1 at least {{macro_f1}} and at least {{baseline_margin}} above the best classical baseline (“always N” gives about 0.24) |
| Minority-class performance | Recall (sensitivity) of V, S and F, with precision and F1 | Recall V at least {{recall_V}}, S at least {{recall_S}}, F at least {{recall_F}} |
| Noise robustness | Macro-F1 under real NSTDB noise (bw, em, ma) at 12, 6 and 0 dB, compared with clean | Drop from clean to 0 dB of at most {{noise_drop_0db}} (mean of the three noise types) |
| Model complexity | Size of the 8-bit model; macro-F1 change from FP32 to INT8 | At most {{size_mb}} MB (goal: 100,000 parameters or fewer); INT8 loses at most {{int8_drop}} macro-F1 |
| MAX78002 compatibility | Only layers supported by the MAX78002; official ai8x synthesis | Synthesis with ai8xize passes (software-level evidence; no board is used) |
| Process quality | Data split and use of the test set | No record in two splits; DS2 used only for the final evaluation and every use logged |
## 1.5 Constraints
- Limited Edge-AI memory
The limit of the Edge-AI memory is 2MB, which means that we will have to fit inside that 2MB constraint. By making the model fit into such constraints, we lose out on the opportunity for an increase in model complexity.
- Fixed-point/INT8 computation
For actual deployment into the MAX78002, the model would have to be quantized down to 8-bit for the deployment.
- Class imbalance
The dataset is skewed, with the N-class making up the majority of the heartbeats, and the F-class making up less than 1%. This prevents the use of standard accuracy metrics.
- Limited number of subjects
The number of records in the database is 48, and the number of subjects is 47. Four records contain paced beats (102, 104, 107, 217) and are excluded, so 44 records from 43 subjects are used. Because there is such a limited number of subjects, the model heavily depends on which subjects end up in the test set, which is why we are using the (recommended) standard protocol. Not to mention that two separate records (201 and 202) are on the same subject. This is something to keep in mind to prevent testing and training on the same subject.
- Noise
Possible noise constraints can be addressed through band-pass filtering during pre-processing.
- Computational resources
Because we will be using a 1D CNN, the device will not require substantial computational power.
- Absence of physical MAX78002 hardware
Without a board, latency, energy use and real RAM use cannot be measured. Deployment feasibility is shown only at software level, with the official MAX78002 toolchain (quantization and synthesis).
## 1.6 Risks and Challenges
The development of an ECG heartbeat classification system presents several challenges related to data quality, model generalization, class imbalance, and hardware limitations. Identifying these risks early is important to ensure that the resulting model is reliable and evaluated fairly.
| Risk | Why is it important | Proposed mitigation |
|---|---|---|
| **Class imbalance** | The four heartbeat classes (N, S, V and F) are not equally represented in the MIT-BIH Arrhythmia Database. Normal heartbeats are much more common than abnormal ones. The model could therefore achieve high overall accuracy while performing poorly on rare but clinically relevant classes. | The proposed mitigation is to try weighted sampling or class weights to help the model learn from the less common heartbeat classes. |
| **Data leakage and poor generalization** | Heartbeats from the same patient can share similar characteristics. If beats from one patient appear in both training and test sets, the model may learn patient-specific patterns rather than generalizable heartbeat characteristics. This can produce overly optimistic performance estimates. | The proposed mitigation is to split the ECG records, so that no patient appears in both training and test sets. |
| **Noise and signal variability** | ECG recordings may contain baseline wander, muscle noise, electrode movement artefacts, and differences in signal amplitude or morphology. These variations can make it more difficult to distinguish between heartbeat classes and may reduce the model’s robustness when processing real-world signals. | The proposed mitigation is to try filtering, normalization and noise augmentation to see if they help the model handle noisy ECG signals. |
| **Hardware and computational limitations** | A model that performs well in Python may be too large or use unsupported operations for deployment on the MAX78002. | The proposed mitigation is to keep the model relatively small and test whether it works with the MAX78002 toolchain. |
These risks are considered throughout the project. In particular, preventing data leakage and evaluating minority-class performance are essential for obtaining meaningful results, while hardware compatibility must be considered alongside predictive performance to meet the project’s Edge-AI objective.
# 2. Data Understanding
## 2.1 Dataset Description
The name of the database is MIT-BIH Arrhythmia Database. The database contains 48 ~30 minute two-channel recordings of 47 subjects (Moody & Mark, 2001). The recordings have a sampling frequency of 360 Hz per channel. Two or more cardiologists annotated each record, with a total of approximately 110000 annotations in the database. The ECG leads that are available in the dataset are MLII and V1-5. We use 44 of the 48 records (43 subjects, 103,724 expert annotations); the four records with paced beats are left out, as is common practice for this task.
## 2.2 Target Classes
| Final class | Original annotations included | Number of beats (all annotations) | Number of beats (final dataset) |
|---|---|---|---|
| N | N, L, R, e, j | 90,125 | 90,076 |
| S | S, A, a, J | 2,781 | 2,781 |
| V | V, E | 7,009 | 7,008 |
| F | F | 803 | 802 |
| **Total** | | **100,718** | **100,667** |
Annotations excluded are the paced beats “/” and “f”, which belong to the paced records that are not used, “Q” (unclassifiable beat, 15 beats) and the non-beat marks + ~ ! " x | [ ] (2,991 marks). The last column is 51 beats lower than the first, because their one-second window would reach outside the recording. All exclusions are counted in a file written by the preparation code.
## 2.3 Exploratory Data Analysis
### Example ECG signal:
![Example ECG signal](eda_ecg_signal.png|0.95)
### The two channels of one record:
![Both channels of record 101](eda_channels.png|0.95)
### Example heartbeat from each class:
![Example heartbeats](eda_example_beats.png|1.0)
### Average beat of each class:
![Average beat per class](eda_average_beat.png|0.7)
### Class distribution:
![Class distribution](eda_class_distribution.png|0.95)
### Subject/record distribution:
![Records](eda_record_distribution.png|0.9)
### Signal amplitude distribution:
![Amplitude](eda_amplitude.png|0.95)
### Differences between recordings:
![Signal spread per record](eda_recording_differences.png|0.95)
### Example of noisy ECG:
![Noisiest and cleanest window](eda_noise_windows.png|0.9)
### The MIT-BIH Noise Stress Test Database at 12, 6 and 0 dB:
![Noise types](eda_nstdb_records.png|0.9)
![Muscle artifact at three noise levels](eda_noise_snr_examples.png|0.9)
## 2.4 Data Quality
The data quality of the MIT-BIH Arrhythmia Database was investigated by examining missing values, extreme amplitudes, corrupted segments, class imbalance, differences between recordings and ECG channels, and noisy regions.
### Missing data
The exploratory data analysis found no NaN values in the selected ECG recordings. However, this check does not rule out all forms of signal loss or recording errors.
### Extreme amplitudes
Since the amplitude varies heavily between records, per-beat min-max normalisation is needed to evaluate the shape of the wave instead of the absolute voltage. The peaks at about ±5 mV in the amplitude histogram come mainly from record 116, which touches the limit of the recorder (977 samples beyond 5 mV).
### Corrupted segments
Certain segments contain noise or signal loss, which is addressed through flat-beat rejection (no flat beats were found in the selected records).
### Class imbalance
The dataset contains many more normal heartbeats than abnormal ones. Normal heartbeats make up approximately 89.5% of the data, while fusion heartbeats make up only 0.8%. This imbalance could make it harder for the model to learn to recognise the less common classes. Therefore, it is important to evaluate how well the model performs on each class, rather than only looking at its overall accuracy.
### Differences between recordings
In one record (114), MLII is the second channel. Amplitudes also heavily vary between records.
### Differences between ECG channels
One of the ECG channels is more sensitive than the other. R-peaks also look different depending on the channel.
![Differences between the two channels](eda_channel_differences.png|0.9)
### Possible noisy regions
There are some noisy regions, which will be taken care of through the Butterworth band-pass filter (0.5-40Hz). Experts marked noise or artefacts in 38 of the 44 records; records 108 and 203 show the strongest baseline drift.
![Noisy regions per record](eda_noise_per_record.png|0.9)
## 2.5 Leakage Analysis
Data leakage could occur in several ways, one of them being if we were to randomly split beats. Beats of the same patient share heart geometry, electrode position, gain, baseline level and often the same abnormal morphology repeated hundreds of times (e.g. one patient has hundreds of near-identical V beats). If beats of one patient are in both train and test, the network can recognise the patient instead of the beat type. This results in test accuracy becoming unrealistically high, and does not predict performance on new patients. Our approach to this is using DS1 (22 records) for training and validation, split by record (18 train / 4 validation). DS2 (22 records) is the final test set, and no record appears in both sets.
We checked the size of this effect with a small experiment in the notebook: the same Random Forest reached a macro-F1 of 0.87 when the beats were split randomly and only 0.33 when the records were split.
![Leakage experiment](leakage_demo.png|0.55)
Leakage could also occur from neighbouring beats overlapping, i.e., a one second window around R overlaps with the windows of the previous and next beat. If we were to randomly split, this can result in almost identical samples in both training and testing.
The database has 48 records but 47 subjects, records 201 (DS1) and 202 (DS2), which are records from the same individual. The standard DS1/DS2 split therefore has one subject in both sets. We keep the standard split because the task requires it, and report it as a known limitation.
Normalisation computed on the whole dataset would also leak test statistics. Here, normalisation is per beat, and class weights are only computed on the training set. Validation must also be split by record, and augmentation is only applied to training data. Noise for training and test comes from different halves of the NSTDB noise records.
<!-- pagebreak -->
# 3. Data Preparation
## 3.1 Record Selection
For the record selection, the commonly used inter-patient MIT-BIH partitioning is used. This means that the records used for training and validation are: 101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, & 230.
For the final test, the records used are: 100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, & 234. The four records with paced beats (102, 104, 107, 217) are not used in any split.
## 3.2 ECG Channel Selection
We are using the MLII channel primarily because it is the standard lead for the MIT-BIH classification, but also for clean preprocessing and an easier detection of the heartbeat. The preprocessing is much easier because the R-peaks are much more prominent in the MLII channel.
Because of a single record, where MLII is the second channel, MLII is instead selected by name. If a record does not have MLII at all, it will fall back to channel 0 and report it.
## 3.3 Beat Extraction
- How R-peak annotations are obtained: they are read directly from the .atr-files.
- Samples before the annotated beat: 180.
- Samples after the annotated beat: 179.
- Final input length: 360 samples.
## 3.4 Signal Preprocessing
Raw ECG Data with MLII lead<br/>↓<br/>Noise injection (only for the noise test, added to the raw signal)<br/>↓<br/>Band-pass filtering (0.5-40Hz, Butterworth, order 4, zero-phase)<br/>↓<br/>Beat segmentation around annotated R-peaks<br/>↓<br/>Drop flat-line beats<br/>↓<br/>Normalization per beat (min-max to [-1, 1])<br/>↓<br/>Model Input
The zero-phase filter uses samples from the future, so it cannot run in real time. A device would need a causal filter; the preparation code has an option for this, but it was not used for the results.
![Preprocessing of one beat per class: raw, band-pass filtered, normalised](eda_preprocessing.png|1.0)
## 3.5 Label Preparation
**N (Normal):** N → N, L → N, R → N, e → N, j → N
**S (Supraventricular):** S → S, a → S, A → S, J → S
**V (Ventricular):** V → V, E → V
**F (Fusion):** F → F
## 3.6 Class Imbalance
The MIT-BIH dataset contains a severe class imbalance. The N class contains the majority of the heartbeats, 90,125, for the total 100,718 heartbeats (90,076 of 100,667 in the final dataset). That is why, with the usage of class weights, forcing the model to prioritize minority classes is a necessity. The weights are computed from the training set only (N 0.28, S 13.15, V 3.23, F 25.51). {{class_weight_sentence}}
The following alternatives were rejected:
- Oversampling: random oversampling would cause overfitting.
- Undersampling: undersampling would starve the model. The model needs a big volume of heartbeats to learn what a healthy baseline looks like.
- Balanced batches: balanced batches would require the same small amount of F beats. This would result in the model learning about those specific patients instead of learning generalizable features. {{sampler_sentence}}
- Focal loss: the class weights provide a more simple and interpretable baseline than focal loss, and without needing additional hyperparameter tuning.
- No balancing: without balancing, the model would predict the N-class for every input, achieving high accuracy while not achieving its objective of detecting dangerous arrhythmias.
## 3.7 Data Augmentation
To prevent the model from overfitting from the limited pool of subjects, and to simulate real-world conditions for the Edge-AI deployment, noise injection and baseline wander injection was used as augmentation on our training data, together with a random gain (0.9–1.1) and a small time shift (up to 3 samples). Augmentation is applied to training beats only. The noise comes from the first half of the NSTDB recordings; the second half is kept for the DS2 noise test.
## 3.8 Final Dataset
| Split | Records | N | S | V | F | Total |
|---|---|---|---|---|---|---|
| Training | 18 | 36,398 | 774 | 3,150 | 399 | 40,721 |
| Validation | 4 | 9,443 | 170 | 638 | 15 | 10,266 |
| Test | 22 | 44,235 | 1,837 | 3,220 | 388 | 49,680 |
| **All** | 44 | 90,076 | 2,781 | 7,008 | 802 | 100,667 |
The three splits have a similar class mix; the validation set contains only 15 F beats, so validation results for F are noisy.
![Class mix of the three splits](split_composition.png|0.85)
## 3.9 Reproducibility
**Software versions:** Python (3.9.13), NumPy (1.26.4), SciPy (1.9.1), pandas (1.4.4), & WFDB (4.1.2). {{software_task2}}
**Random seed:** 42.
**Preprocessing parameters:**
- Target lead: MLII
- Filter: Butterworth band-pass (lowcut Hz: 0.5, highcut Hz: 40.0, order: 4, zero phase: true).
- 180 samples before R, 179 samples after R.
- Normalization: Min-max.
- Pipeline order: Noise injection (noise test only), band-pass filter, optional clipping, segmentation, flat-beat rejection, per-beat normalisation.
**Sampling frequency:** 360Hz.
**Input size:** 360 samples.
## 3.10 Summary, Limitations and Next Steps
### What we learned about the data:
- The data is clean (no missing values, no flat parts), but it has a strong class imbalance and large differences between records.
- The lead order is not the same in all records; selecting MLII by name solves this.
- Experts marked noise or artifacts in 38 of 44 records, and some records have strong baseline drift.
- A random beat split gives far too optimistic results.
### Limitations
- Records 201 (train) and 202 (test) are the same person.
- Only 15 F beats in the validation set; validation metrics for F and S are noisy.
- Beats near expert-marked noisy regions are kept in the data.
- R-peak positions come from expert annotations, not from an automatic detector.
- The zero-phase filter cannot be used in real time, and one second of ECG contains no information about the time between beats (RR interval).
- No claim of clinical validity.
### Next steps (CRISP-DM stage 4: Modeling)
- Build a simple baseline and a compact 1D CNN that fits the MAX78002 rules.
- Train with class weights (and compare with the balanced sampler), using the training set; select with the validation set.
- Report per-class results; run the noise test at 12, 6 and 0 dB; compare FP32 and INT8.
@@TASK2@@
<!-- pagebreak -->
# References
Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
Ansari, Y., Mourad, O., Qaraqe, K., & Serpedin, E. (2023). Deep learning for ECG Arrhythmia detection and classification: an overview of progress for period 2017–2023. Frontiers in Physiology, 14, article 1246746. DOI: 10.3389/fphys.2023.1246746
Niu, Y., Wang, H., Wang, H., Zhang, H., Jin, Z., & Guo, Y. (2023). Diagnostic validation of smart wearable device embedded with single-lead electrocardiogram for arrhythmia detection. Digital health, 9, 20552076231198682. https://doi.org/10.1177/20552076231198682
Ramkumar, S., Nerlekar, N., D'Souza, D., Pol, D. J., Kalman, J. M., & Marwick, T. H. (2018). Atrial fibrillation detection using single lead portable electrocardiographic monitoring: a systematic review and meta-analysis. BMJ open, 8(9), e024178. https://doi.org/10.1136/bmjopen-2018-024178
Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. Ch., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220.
Moody, G. B., Muldrow, W. E., & Mark, R. G. (1984). A noise stress test for arrhythmia detectors. Computers in Cardiology, 11, 381–384.
<!-- pagebreak -->
# Appendix A: Changes compared with the first submission
@@CHANGES@@
