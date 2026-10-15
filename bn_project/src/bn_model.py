"""
Member 2 — Bayesian Network for single-reading occupancy estimation.

Network structure (a "naive Bayes" Bayesian Network — one cause node,
several conditionally-independent effect nodes):

                        Occupancy (S0 / S1 / S2)
                       /      |      |      \\
                    CO2     Light   Sound    PIR
                 (Low/High) (Low/High)(Low/High)(0/1)

This encodes the causal story behind the sensors: the number of people in
the room (the hidden cause) raises CO2, is more likely to coincide with
light being on, raises ambient sound, and triggers motion — and, crucially,
GIVEN the true occupancy state, the four sensor readings are assumed
conditionally independent of each other (that's the Bayesian-network
"naive" structure, and the modelling assumption this technique is built
on, same as a standard Naive Bayes classifier).

This is deliberately the complementary technique to Member 1's HMM:
  - HMM:  sequence of readings over time -> exploits temporal structure,
          chronological train/test split, Viterbi/Forward.
  - BN:   ONE reading at ONE instant -> no memory of previous readings,
          random (not chronological) train/test split, exact inference
          by enumeration (trivial here because of the naive-Bayes shape).

Pipeline:
  1. Discretize sensors into Low/High using TRAIN-only thresholds (no
     data leakage) -- same approach as Member 1's HMM, for comparability.
  2. Learn P(Occupancy) (the prior) and P(sensor | Occupancy) for each of
     the 4 sensors from training counts, with Laplace (+1) smoothing.
  3. For each test row, compute the posterior P(Occupancy | all 4 sensor
     readings) via Bayes' rule:
         P(S | co2,light,sound,pir) ∝ P(S) * P(co2|S) * P(light|S) * P(sound|S) * P(pir|S)
     normalized over the 3 states.
  4. Predict the state with the highest posterior; evaluate against the
     real labels (accuracy, macro F1, confusion matrix).
  5. Export per-row posteriors to bn_predictions.csv for Member 3's MDP.

Usage:
    python bn_model.py --input ../output/clean_data_bn.csv --train-frac 0.8
"""

import argparse
import json
import time
import numpy as np
import pandas as pd

STATES = ["S0", "S1", "S2"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
SENSORS = ["co2", "light", "sound", "pir"]


# ---------------------------------------------------------------------------
# Discretization (train-only thresholds, applied to both splits)
# ---------------------------------------------------------------------------
def fit_thresholds(train_df: pd.DataFrame) -> dict:
    """Low/High split point for each continuous sensor = TRAIN median.
    PIR is already binary (0/1) so it needs no threshold."""
    return {
        "co2": train_df["co2"].median(),
        "light": train_df["light"].median(),
        "sound": train_df["sound"].median(),
    }


def discretize(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    for col in ["co2", "light", "sound"]:
        out[col + "_bin"] = np.where(out[col] > thresholds[col], "High", "Low")
    out["pir_bin"] = np.where(out["pir"] > 0, "High", "Low")   # 1 = motion -> High
    return out


# ---------------------------------------------------------------------------
# Learning: prior P(Occupancy) and CPTs P(sensor_bin | Occupancy)
# ---------------------------------------------------------------------------
def estimate_priors(train_df: pd.DataFrame, alpha: float = 1.0) -> dict:
    counts = train_df["true_state"].value_counts().reindex(STATES, fill_value=0).astype(float)
    counts += alpha
    probs = counts / counts.sum()
    return probs.to_dict()


def estimate_cpts(train_df: pd.DataFrame, alpha: float = 1.0) -> dict:
    """cpts[sensor][state][bin_value] = P(sensor=bin_value | Occupancy=state),
    Laplace-smoothed so no probability is ever exactly zero (a single unseen
    combination in training must never kill a whole posterior to zero at
    test time)."""
    cpts = {}
    for sensor in SENSORS:
        col = sensor + "_bin"
        bins = ["Low", "High"]
        table = {}
        for state in STATES:
            sub = train_df[train_df["true_state"] == state]
            counts = sub[col].value_counts().reindex(bins, fill_value=0).astype(float)
            counts += alpha
            probs = counts / counts.sum()
            table[state] = probs.to_dict()
        cpts[sensor] = table
    return cpts


# ---------------------------------------------------------------------------
# Inference: exact posterior by enumeration (trivial given the naive-Bayes
# / conditionally-independent-children structure -- no variable elimination
# machinery needed, but this is still exact Bayesian-network inference,
# not an approximation).
# ---------------------------------------------------------------------------
def posterior(priors: dict, cpts: dict, evidence: dict, states=STATES) -> dict:
    """
    priors:   {state: P(state)}
    cpts:     {variable: {state: {value: P(value|state)}}}
    evidence: {variable: observed_value}   (may be a subset of cpts' keys --
               a variable with no sensor reading is simply marginalized out,
               which for a naive-Bayes-shaped network means it just doesn't
               appear in the product at all)
    Returns:  {state: P(state | evidence)}, normalized to sum to 1.
    """
    unnorm = {}
    for s in states:
        p = priors[s]
        for var, val in evidence.items():
            p *= cpts[var][s][val]
        unnorm[s] = p
    total = sum(unnorm.values())
    if total == 0:
        # Should not happen with Laplace smoothing, but guard anyway.
        return {s: 1.0 / len(states) for s in states}
    return {s: unnorm[s] / total for s in states}


def predict_state(priors: dict, cpts: dict, evidence: dict, states=STATES) -> str:
    post = posterior(priors, cpts, evidence, states)
    return max(post, key=post.get)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def macro_f1(y_true, y_pred, states=STATES):
    f1s = {}
    for s in states:
        tp = sum((t == s and p == s) for t, p in zip(y_true, y_pred))
        fp = sum((t != s and p == s) for t, p in zip(y_true, y_pred))
        fn = sum((t == s and p != s) for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s[s] = f1
    return sum(f1s.values()) / len(states), f1s


def confusion_matrix(y_true, y_pred, states=STATES):
    cm = pd.DataFrame(0, index=states, columns=states)
    for t, p in zip(y_true, y_pred):
        cm.loc[t, p] += 1
    return cm


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../output/clean_data_bn.csv")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pred-out", default="../output/bn_predictions.csv")
    ap.add_argument("--metrics-out", default="../output/bn_metrics.txt")
    ap.add_argument("--params-out", default="../output/bn_params.json")
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    print(f"Loaded {len(df)} rows.")

    # --- RANDOM stratified split (NOT chronological) -----------------------
    # This is the key structural difference from Member 1's HMM: the BN
    # treats every row as an independent snapshot, with no sequence to
    # preserve, so a random shuffle is the statistically correct split here
    # (and it avoids the HMM's "zero S1 rows in the test window" problem,
    # since a stratified shuffle guarantees every class appears in test).
    rng = np.random.RandomState(args.seed)
    train_parts, test_parts = [], []
    for state in STATES:
        sub = df[df["true_state"] == state].sample(frac=1.0, random_state=rng)
        n_train = int(len(sub) * args.train_frac)
        train_parts.append(sub.iloc[:n_train])
        test_parts.append(sub.iloc[n_train:])
    train_df = pd.concat(train_parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    test_df = pd.concat(test_parts).sample(frac=1.0, random_state=rng).reset_index(drop=True)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows (stratified random split)")
    print("Train class balance:\n", train_df["true_state"].value_counts().reindex(STATES))
    print("Test class balance:\n", test_df["true_state"].value_counts().reindex(STATES))

    # --- Discretize ----------------------------------------------------
    thresholds = fit_thresholds(train_df)
    print(f"\nThresholds (train-only): {thresholds}")
    train_df = discretize(train_df, thresholds)
    test_df = discretize(test_df, thresholds)

    # --- Learn prior + CPTs ---------------------------------------------
    priors = estimate_priors(train_df)
    cpts = estimate_cpts(train_df)
    print("\nLearned prior P(Occupancy):")
    for s in STATES:
        print(f"  P({s}) = {priors[s]:.4f}")
    print("\nLearned CPTs P(sensor=bin | Occupancy):")
    for sensor in SENSORS:
        print(f"  {sensor}:")
        for s in STATES:
            print(f"    {s}: {cpts[sensor][s]}")

    # --- Inference on test set ------------------------------------------
    t0 = time.time()
    records = []
    for _, row in test_df.iterrows():
        evidence = {sensor: row[sensor + "_bin"] for sensor in SENSORS}
        post = posterior(priors, cpts, evidence)
        pred = max(post, key=post.get)
        records.append({
            "timestamp": row["timestamp"],
            "predicted_state": pred,
            "P(S0)": post["S0"], "P(S1)": post["S1"], "P(S2)": post["S2"],
            "true_state": row["true_state"],
        })
    inference_time = time.time() - t0
    pred_df = pd.DataFrame(records)

    # --- Evaluate ---------------------------------------------------------
    y_true = pred_df["true_state"].tolist()
    y_pred = pred_df["predicted_state"].tolist()
    accuracy = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
    mf1, per_class_f1 = macro_f1(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n=== Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {mf1:.4f}")
    print(f"Per-class F1: {per_class_f1}")
    print(f"Inference time ({len(test_df)} rows): {inference_time:.4f}s")
    print("\nConfusion matrix (rows=true, cols=predicted):")
    print(cm)

    # --- Write outputs ------------------------------------------------
    pred_df.to_csv(args.pred_out, index=False)
    print(f"\nWrote predictions to {args.pred_out}")

    with open(args.metrics_out, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {mf1:.4f}\n")
        f.write(f"Per-class F1: {per_class_f1}\n")
        f.write(f"Inference time (s, {len(test_df)} rows): {inference_time:.4f}\n")
        f.write(f"Test set size: {len(test_df)}\n\n")
        f.write("Confusion matrix (rows=true, cols=predicted):\n")
        f.write(str(cm) + "\n")
    print(f"Wrote metrics to {args.metrics_out}")

    params = {
        "priors": priors,
        "cpts": cpts,
        "thresholds": thresholds,
        "states": STATES,
        "sensors": SENSORS,
    }
    with open(args.params_out, "w") as f:
        json.dump(params, f, indent=2, default=float)
    print(f"Wrote model parameters to {args.params_out}")


if __name__ == "__main__":
    main()
