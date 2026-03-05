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

### 2.2 Sampling Rate Harmonisation

Both devices recorded at approximately 100 Hz (10 ms intervals). Minor timing inconsistencies between accelerometer and gyroscope streams were resolved using `merge_asof` with a 15 ms tolerance, followed by linear interpolation onto a fixed 10 ms grid. This brings all recordings to exactly 100 Hz, ensuring that window sizes and extracted features are directly comparable across participants and sessions.

### 2.3 Windowing Strategy

- **Window size:** 50 samples (0.5 seconds) with 50% overlap (25-sample step).
- At 100 Hz, a 0.5-second window captures at least one full stride cycle during walking (~2 Hz cadence) and one jump cycle (~1 Hz), while remaining short enough to detect activity transitions without blurring them across window boundaries.

---

## 3. Feature Extraction

Seven features are extracted from each window, computed from the scalar acceleration magnitude |a| = √(ax² + ay² + az²). This magnitude is orientation-independent, reducing sensitivity to how the phone is held. All features are **Z-score normalised** (StandardScaler) so that features on different scales contribute equally to the HMM emission probabilities.
| Feature | Domain | Why It Distinguishes Activities |
|--------------------|----------|-----------------------------------------------------------|
| RMS | Time | Motion intensity (still low, jumping very high) |
| Standard Deviation | Time | Variability (still near 0, dynamic activities higher) |
| SMA | Time | Total movement magnitude (orientation-independent) |
| Zero Crossing Rate | Time | Oscillation rate (walking cyclic, still flat) |
| Dominant Frequency | Frequency (FFT) | Walking ~2 Hz, jumping ~1 Hz, still near 0 Hz |
| Spectral Energy | Frequency (FFT) | Total spectral power (very high for jumping) |
| Spectral Entropy | Frequency (FFT) | Spectrum complexity (walking periodic, still noisy) |

---

## 4. HMM Setup and Implementation

### 4.1 Model Components

| Component              | Symbol | Type       | Value / Method                              |
| ---------------------- | ------ | ---------- | ------------------------------------------- |
| Hidden States          | Z      | Discrete   | 4: still, standing, walking, jumping        |
| Observations           | X      | Continuous | 7-feature vector per 0.5s window            |
| Transition Matrix      | A      | 4×4 matrix | Uniform init, learned via Baum–Welch        |
| Emission Probabilities | B      | Gaussian   | Diagonal covariance, learned via Baum–Welch |
| Initial Probabilities  | π      | 4-vector   | Uniform init, learned via Baum–Welch        |

### 4.2 Baum–Welch Training

Each model is trained using the Baum–Welch algorithm (Expectation–Maximisation) via `hmmlearn`, with 4 hidden states per model, diagonal covariance matrices, convergence threshold ε = 1e-4 (training stops when the log-likelihood improvement between iterations falls below ε), and a maximum of 200 iterations. All four models converged; walking required more iterations due to its feature similarity with standing.

### 4.3 Viterbi Decoding

## The Viterbi algorithm uses dynamic programming to find the single most likely hidden state sequence for a given observation sequence. Classification proceeds by: (1) scoring the test sequence against all four models, (2) selecting the highest log-likelihood activity, and (3) running Viterbi on the winning model to decode the internal state path.
