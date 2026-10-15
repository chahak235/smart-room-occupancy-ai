# Smart Room Occupancy Estimation — HMM Module

**AI-Based Smart Room Occupancy Estimation and Energy Management Using Hidden Markov Models, Bayesian Networks, and Markov Decision Processes**

This repository holds the Hidden Markov Model (HMM) component of a 3-person AI course project. The full system estimates how many people are in a room from environmental sensor data, then recommends an energy-management action (lights/AC) under uncertainty. The project is split across three techniques, each solving a different part of the problem:

| Member | Technique | Role |
|---|---|---|
| **1 (this repo)** | **Hidden Markov Model** | Estimate occupancy state from a sequence of sensor readings |
| 2 | Bayesian Network | Estimate occupancy probability from a single sensor reading |
| 3 | MDP + Value/Policy Iteration | Choose the optimal energy-management action |

## Problem

Conventional room energy-management systems run on fixed schedules or simple motion-sensor rules, wasting electricity on empty rooms or under-serving occupied ones. This module estimates the room's true occupancy state from noisy, indirect sensor evidence — the room's actual occupancy is never observed directly, only inferred from how CO₂, light, sound, and motion sensors behave.

## Dataset

**[UCI Room Occupancy Estimation Dataset](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)**
- 10,129 time-ordered observations, collected every ~30 seconds over 4 days
- Sensors: temperature, light, sound, CO₂, and 2 PIR motion sensors
- Ground truth: actual occupant count (0–3 people) per row

The raw count is mapped into 3 states for this project:

| State | Meaning |
|---|---|
| S0 | Empty (0 people) |
| S1 | Low occupancy (1 person) |
| S2 | Higher occupancy (2–3 people) |

## What's in this repo

```
data/
  Occupancy_Estimation.csv   ← raw UCI dataset
src/
  preprocess.py              ← cleans raw data, builds timestamp, maps states, aggregates sensors
  hmm_model.py                ← trains the HMM, runs Forward + Viterbi, evaluates, exports predictions
  test_toy_example.py         ← sanity-checks Forward/Viterbi against a hand-worked example
  make_visuals.py             ← generates all result charts
output/
  clean_data.csv               ← preprocessed dataset
  hmm_predictions.csv          ← Viterbi-decoded test predictions (hand-off format for Member 3)
  hmm_metrics.txt              ← accuracy / F1 / confusion matrix summary
  confusion_matrix.png
  transition_matrix.png
  state_distribution.png
  occupancy_timeline.png
  state_probabilities.png
```

## How the pipeline works

1. **`preprocess.py`** — combines Date + Time into one timestamp, sorts chronologically, maps the raw occupant count to S0/S1/S2, and averages the multi-sensor readings (4 light sensors → 1, 4 sound sensors → 1, 2 PIR sensors → 1).

2. **`hmm_model.py`** — the core model:
   - Splits data 80/20 **chronologically** (no shuffling — order matters for an HMM)
   - Discretizes each sensor into Low/High, with thresholds fit on the training split only (no data leakage)
   - Estimates the transition matrix (A), emission probabilities (B), and initial distribution (π) from the training data by counting, with Laplace smoothing
   - Runs the **Viterbi algorithm** to decode the most likely occupancy sequence on the test data
   - Runs the **Forward algorithm** to compute sequence likelihood and per-step belief probabilities
   - Evaluates against the real labels: accuracy, macro F1, confusion matrix, log-likelihood, inference time
   - Exports predictions to `hmm_predictions.csv` for Member 3's MDP

3. **`test_toy_example.py`** — before trusting the model on real data, this verifies the Forward and Viterbi implementations against a small hand-solvable example with a known correct answer.

4. **`make_visuals.py`** — turns the results into charts for the report.

## Running it

```bash
cd src
python preprocess.py --input ../data/Occupancy_Estimation.csv --output ../output/clean_data.csv
python test_toy_example.py
python hmm_model.py --input ../output/clean_data.csv --train-frac 0.8
python make_visuals.py
```

Requires `pandas`, `numpy`, and `matplotlib`:
```bash
pip install pandas numpy matplotlib
```

## Results

| Metric | Value |
|---|---|
| Accuracy | 99.90% |
| Macro F1 | 0.665 |
| Log-likelihood (test) | -2859.92 |
| Viterbi inference time | ~26 ms / 2026 steps |

**Note on macro F1:** accuracy alone is misleading here. The chronological test split (the dataset's final ~18 hours) happens to contain zero S1 (1-person) observations — all 459 S1 rows fall inside the training window — so the model was never tested on that class. Macro F1 surfaces this; raw accuracy hides it. This is a real property of a non-stationary, imbalanced dataset split chronologically, not a bug.

See `output/state_distribution.png` for the train/test class balance and `output/confusion_matrix.png` for the full breakdown.

## Live demo

An interactive simulator built on these exact trained parameters (A, B, π) lets you add people to the room and watch the HMM's belief update step by step, with an energy-action recommendation that follows: **[Smart Room Simulator](https://claude.ai/artifact/RxWiHsqHH1QhGJQYvhuetd)**

## License / attribution

Dataset: UCI Machine Learning Repository, [Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation).