# Smart Room Occupancy Estimation — Bayesian Network Module

**AI-Based Smart Room Occupancy Estimation and Energy Management Using Hidden Markov Models, Bayesian Networks, and Markov Decision Processes**

This repository holds the Bayesian Network (BN) component of a 3-person AI course project. The full system estimates how many people are in a room from environmental sensor data, then recommends an energy-management action (lights/AC) under uncertainty. The project is split across three techniques, each solving a different part of the problem:

| Member | Technique | Role |
|---|---|---|
| 1 | Hidden Markov Model | Estimate occupancy state from a *sequence* of sensor readings |
| **2 (this repo)** | **Bayesian Network** | Estimate occupancy probability from a *single* sensor reading |
| 3 | MDP + Value/Policy Iteration | Choose the optimal energy-management action |

## Problem

Member 1's HMM exploits the fact that occupancy doesn't change instantly — it uses the *previous* readings to help interpret the *current* one. This module answers a different question: **if all you have is one snapshot of sensor readings, right now, with no history at all, what's the best estimate of how many people are in the room?**

That's a realistic constraint in its own right — a newly-booted sensor node, a system that just came back online after an outage, or simply a system design that wants a fast, memoryless estimate it can cross-check against the HMM's sequence-based one. A Bayesian Network is the right tool for exactly this: a probabilistic model of how one hidden cause (occupancy) produces several observable effects (CO₂, light, sound, motion) at a single instant, with no notion of "the previous time step" built in.

## Dataset

**[UCI Room Occupancy Estimation Dataset](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation)** — the same dataset Member 1 used, so both techniques' results are directly comparable.
- 10,129 time-ordered observations, collected every ~30 seconds over 4 days
- Sensors: temperature, light, sound, CO₂, and 2 PIR motion sensors
- Ground truth: actual occupant count (0–3 people) per row

The raw count is mapped into the same 3 states used throughout the project:

| State | Meaning |
|---|---|
| S0 | Empty (0 people) |
| S1 | Low occupancy (1 person) |
| S2 | Higher occupancy (2–3 people) |

## The network structure

```
                        Occupancy (S0 / S1 / S2)
                       /      |      |      \
                    CO2     Light   Sound    PIR
```

This is a **naive-Bayes-shaped Bayesian Network**: one hidden cause node (Occupancy) with four observed child nodes (CO₂, Light, Sound, PIR), each assumed **conditionally independent of the others given the true occupancy state**. That's the standard, textbook Bayesian-network encoding of "one cause, several effects" — more people in the room independently pushes CO₂ up, light more likely to be on, ambient sound up, and motion sensors to fire, and none of the sensors are assumed to influence each other directly, only through the shared cause.

Because of this shape, exact inference doesn't need variable elimination or belief propagation — it collapses to a direct application of Bayes' rule:

```
P(Occupancy | CO2, Light, Sound, PIR)  ∝  P(Occupancy) · P(CO2|Occupancy) · P(Light|Occupancy) · P(Sound|Occupancy) · P(PIR|Occupancy)
```

normalized over the 3 states. This is still **exact** Bayesian-network inference (not an approximation) — it's exact *because* of the structure, the same shortcut a Naive Bayes classifier relies on.

## What's in this repo

```
data/
  Occupancy_Estimation.csv    ← raw UCI dataset (same file Member 1 uses)
src/
  preprocess_bn.py            ← cleans raw data, maps states, aggregates sensors
  bn_model.py                 ← learns the network's CPTs, runs inference, evaluates, exports predictions
  test_toy_example_bn.py      ← sanity-checks the posterior() function against a hand-worked example
  make_visuals_bn.py          ← generates all result charts
output/
  clean_data_bn.csv           ← preprocessed dataset
  bn_predictions.csv          ← per-row posterior predictions (hand-off format for Member 3)
  bn_metrics.txt              ← accuracy / F1 / confusion matrix summary
  bn_params.json              ← learned priors, CPTs, and discretization thresholds
  network_structure.png
  confusion_matrix_bn.png
  cpt_heatmaps.png
  prior_distribution.png
  posterior_examples.png
```

## How the pipeline works

1. **`preprocess_bn.py`** — maps the raw occupant count to S0/S1/S2 and averages the multi-sensor readings (4 light sensors → 1, 4 sound sensors → 1, 2 PIR sensors → 1), identically to Member 1's aggregation so both models see the same signals. Unlike the HMM's preprocessing, **row order doesn't matter here** — each row is treated as an independent snapshot, not a step in a sequence.

2. **`bn_model.py`** — the core model:
   - Splits data 80/20 with a **random stratified shuffle** (not chronological — this is the key structural difference from the HMM, see below)
   - Discretizes CO₂/Light/Sound into Low/High, with thresholds fit on the training split only (no data leakage); PIR is already binary
   - Learns the prior **P(Occupancy)** and the four **CPTs P(sensor | Occupancy)** from training counts, with Laplace (+1) smoothing so no probability is ever exactly zero
   - For every test row, computes the exact posterior **P(Occupancy | all 4 readings)** via Bayes' rule and predicts the most probable state (MAP estimate)
   - Evaluates against the real labels: accuracy, macro F1, confusion matrix, inference time
   - Exports every row's full posterior distribution to `bn_predictions.csv` for Member 3's MDP

3. **`test_toy_example_bn.py`** — before trusting the model on real data, this verifies the `posterior()` function against a small hand-solvable 2-state network (a classic "Weather → Umbrella, Mood" example) with Bayes' rule worked out by hand, and checks the output matches to 8 decimal places.

4. **`make_visuals_bn.py`** — turns the results into charts for the report: the network diagram, the learned CPTs as heatmaps, the confusion matrix, prior-vs-actual class balance, and example posteriors on real test rows.

## Why a random split here, but chronological for the HMM

This is the single most important modelling choice in this module, and it's worth stating explicitly because it's easy to get backwards:

- The **HMM** models a *sequence* — training and testing on shuffled rows would let it "see the future" through adjacent time steps, so it must be split **chronologically**.
- The **Bayesian Network** models *independent snapshots* — it has no notion of sequence at all, so a chronological split would just be an arbitrary, avoidable source of extra variance. A **random stratified split** is the statistically correct choice here, and it has a side benefit: every class is guaranteed to appear in both train and test (unlike the HMM's chronological split, where the test window happened to contain zero S1 rows — see Member 1's README).

## Running it

```bash
cd src
python preprocess_bn.py --input ../data/Occupancy_Estimation.csv --output ../output/clean_data_bn.csv
python test_toy_example_bn.py
python bn_model.py --input ../output/clean_data_bn.csv --train-frac 0.8
python make_visuals_bn.py
```

Requires `pandas`, `numpy`, and `matplotlib`:
```bash
pip install pandas numpy matplotlib
```

## Results

| Metric | Value |
|---|---|
| Accuracy | 94.62% |
| Macro F1 | 0.6489 |
| Per-class F1 | S0: 0.990, S1: 0.101, S2: 0.856 |
| Inference time | ~60 ms / 2,027 rows |

**Note on the S1 class:** unlike the HMM's test split (which had zero S1 rows), this random split gives S1 a fair shot — 92 S1 rows in test — and the BN still struggles badly on it (F1 = 0.10, only 5/92 correctly identified, most misclassified as S2). Looking at the learned CPTs (`cpt_heatmaps.png`) explains why: at the Low/High discretization level used here, S1's sensor signature (CO₂ High 93%, Light High 100%, Sound High 84%) is nearly indistinguishable from S2's (CO₂ High 100%, Light High 100%, Sound High 98%) — the only sensor that meaningfully separates them is PIR (37% High for S1 vs. 71% High for S2), and that's not enough on its own to reliably tell "1 person" from "2–3 people" from a single instant's readings. This is a genuine limitation of coarse 2-bin discretization combined with the conditional-independence assumption, not a bug — it's a real, reportable finding about where single-reading inference hits a ceiling that sequence-based inference (Member 1's HMM) can push past by using temporal context.

See `output/cpt_heatmaps.png` for the learned conditional probabilities and `output/confusion_matrix_bn.png` for the full breakdown.

## Hand-off

**For Member 3 (MDP):** `output/bn_predictions.csv` has one row per test instance with columns `timestamp, predicted_state, P(S0), P(S1), P(S2), true_state` — the full posterior belief, not just the MAP prediction, so the MDP can use the uncertainty directly (e.g. don't commit to an aggressive energy-saving action when P(S0) is only 0.6). This is the same output shape as Member 1's `hmm_predictions.csv`, so Member 3 can take either (or both, to compare) as an input belief state.

**For Member 2 (me) — still open:**
- Coordinate with Member 3 on whether they want the HMM's or the BN's posteriors (or an ensemble of both) as the MDP's belief-state input
- Possibly extend the discretization beyond Low/High (e.g. Low/Medium/High) to see if it fixes the S1-vs-S2 confusion — flagged as a limitation above, not yet implemented
- Add the toy-example and real-result numbers to the shared project report alongside Member 1's HMM section

## Where everything lives

```
bn_project/
  README.md
  data/Occupancy_Estimation.csv
  src/{preprocess_bn.py, bn_model.py, test_toy_example_bn.py, make_visuals_bn.py}
  output/{clean_data_bn.csv, bn_predictions.csv, bn_metrics.txt, bn_params.json, *.png}
```

Reproduce the whole pipeline with the four commands under **Running it** above, in order.

## License / attribution

Dataset: UCI Machine Learning Repository, [Room Occupancy Estimation](https://archive.ics.uci.edu/dataset/864/room+occupancy+estimation).
