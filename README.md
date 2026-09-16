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
