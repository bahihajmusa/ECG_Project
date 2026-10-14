# Edge-Ready ECG Classification

**Project:** Edge-Ready ECG Classification using CRISP-DM  
**Course:** Data Science / Edge AI  
**Date:** 2026  

This repository contains Task 1 of the course project, covering the first three stages of the CRISP-DM methodology:

1. **Business/Application Understanding**
2. **Data Understanding**
3. **Data Preparation**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)


---

## Project Overview

This project develops a compact machine learning system for ECG heartbeat classification on the Analog Devices MAX78002 Edge-AI microcontroller. The system classifies heartbeats into four categories:

| Class | Description | Example Conditions |
|-------|-------------|-------------------|
| **N** | Normal beats | Normal sinus rhythm, bundle branch blocks |
| **S** | Supraventricular ectopic beats | Atrial premature beats, junctional beats |
| **V** | Ventricular ectopic beats | Premature ventricular contractions |
| **F** | Fusion beats | Fusion of ventricular and normal beats |

### Why Edge-AI?

- **Privacy:** Patient data stays on the device
- **Latency:** Real-time classification without cloud communication
- **Reliability:** Works without internet connectivity
- **Power efficiency:** Low energy consumption for continuous monitoring

---

## Repository Structure

```
ECG_Project/
|
|-- README.md                          # This file
|-- requirements.txt                   # Python dependencies
|
|-- notebooks/
|   └── EDA.ipynb                      # Exploratory Data Analysis
|
|-- src/
|   ├── prepare_data.py                # Main data preparation script
|   └── dataset.py                     # PyTorch Dataset class
|
|-- configs/                           # Configuration files
|
|-- results/                           # Generated statistics and outputs
|
|-- data/                              # Data directory (not in git)
|   ├── raw/                           # Downloaded WFDB files
|   └── processed/                     # Processed NumPy arrays
```

## Reproducibility record

  random_seed: 42
  software_versions: {'python': '3.9.13', 'numpy': '1.26.4', 'scipy': '1.9.1', 'pandas': '1.4.4', 'wfdb': '4.1.2'}
  sampling_frequency: 360
  input_length: 360
  samples_before_r: 180
  samples_after_r: 179
  target_lead: MLII
  filter: {'type': 'Butterworth band-pass (SOS)', 'lowcut_hz': 0.5, 'highcut_hz': 40.0, 'order': 4, 'zero_phase': True}
  normalization_method: minmax
  clip_sigma: None
  pipeline_order: ['(noise injection, noisy test only)', 'band-pass filter', 'optional clipping', 'segmentation', 'flat-beat rejection', 'per-beat normalisation']