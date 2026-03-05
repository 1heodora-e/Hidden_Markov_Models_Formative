# Modelling Human Activity States Using Hidden Markov Models

**Machine Learning Techniques II — Formative 2**  
**Cohort 1 | Team 7**
Victoria Fakunle | Theodora Egbunike  
March 4, 2026
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/1heodora-e/Hidden_Markov_Models_Formative/blob/main/HMM_Activity_Recognition_Notebook.ipynb)

---

## 1. Background and Motivation

Falls and prolonged sedentary behaviour among elderly individuals represent a major public health concern, particularly in settings where continuous clinical monitoring is not feasible. Our group's use case focuses on **passive activity monitoring for elderly care**: a smartphone carried by an elderly person continuously recognises their movement states (still, standing, walking, or jumping), enabling caregivers to be alerted to prolonged inactivity or sudden high-impact events.
Hidden Markov Models are well-suited to this task because human activity evolves as a temporal sequence of hidden states, each producing observable sensor readings. By jointly modelling the temporal transitions between activities and the probabilistic sensor emissions within each activity, an HMM can robustly infer the most likely activity sequence from a continuous, noisy stream of accelerometer and gyroscope data without requiring network connectivity or manual annotation.

---

## 2. Data Collection and Preprocessing

### 2.1 Dataset Overview

Data were collected using the **Sensor Logger app (iOS)** by both group members across two separate recording sessions: a training session and a held-out test session. Each participant recorded four activities: **still** (phone on flat surface), **standing**, **walking**, and **jumping**, for 5–10 seconds per trial. This yielded **50 labelled training files** and **16 unseen test files**.
| Participant | Device (OS) | Sampling Rate | Activities | Files |
|-------------------|-------------------|---------------|------------|--------------------|
| Victoria Fakunle | iPhone 12 Pro (iOS) | ~100 Hz | 4 | 25 train + 8 test |
| Theodora Egbunike | iPhone 12 Pro Max (iOS) | ~100 Hz | 4 | 25 train + 8 test |
| **Total** | — | 100 Hz (harmonised) | 4 | 50 train + 16 test |
