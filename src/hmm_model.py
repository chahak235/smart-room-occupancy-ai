"""
Member 1 — HMM for Smart Room Occupancy Estimation.

Implements, from scratch (no hmmlearn in the submitted path):
  - train/test split (chronological, no shuffling)
  - discretization of CO2/Light/Sound into Low/High bins fit on TRAIN only
  - PIR already binary (No motion / Motion)
  - transition matrix A, emission tables B, initial distribution pi,
    all estimated by counting + Laplace smoothing
  - Forward algorithm (sequence likelihood / filtering)
  - Viterbi algorithm (most likely state sequence -> main prediction)
  - evaluation: accuracy, macro F1, confusion matrix, log-likelihood,
    inference time
  - export of predictions to CSV in the hand-off format the team agreed on

States:  S0 = Empty, S1 = Low occupancy, S2 = Higher occupancy
Observations: a 4-tuple (co2_bin, pir_bin, sound_bin, light_bin), each
              binary, giving 16 possible observation symbols.

Usage:
    python hmm_model.py --input ../output/clean_data.csv --train-frac 0.8
"""

import argparse
import time
import itertools
import numpy as np
import pandas as pd

STATES = ["S0", "S1", "S2"]
STATE_IDX = {s: i for i, s in enumerate(STATES)}
N_STATES = len(STATES)

# Observation symbol = (co2_bin, pir_bin, sound_bin, light_bin), each in {0,1}
# 0 = Low/No-motion, 1 = High/Motion
OBS_SYMBOLS = list(itertools.product([0, 1], repeat=4))
OBS_IDX = {sym: i for i, sym in enumerate(OBS_SYMBOLS)}
N_OBS = len(OBS_SYMBOLS)


# ---------------------------------------------------------------------------
# Discretization (thresholds fit on TRAIN split only, applied to both)
# ---------------------------------------------------------------------------
def fit_thresholds(train_df: pd.DataFrame) -> dict:
    return {
        "co2": train_df["co2"].median(),
        "sound": train_df["sound"].median(),
        "light": train_df["light"].median(),
    }


def discretize(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = df.copy()
    out["co2_bin"] = (out["co2"] > thresholds["co2"]).astype(int)
    out["sound_bin"] = (out["sound"] > thresholds["sound"]).astype(int)
    out["light_bin"] = (out["light"] > thresholds["light"]).astype(int)
    out["pir_bin"] = (out["pir"] > 0).astype(int)  # already ~binary
    out["obs_symbol"] = list(
        zip(out["co2_bin"], out["pir_bin"], out["sound_bin"], out["light_bin"])
    )
    out["obs_idx"] = out["obs_symbol"].map(OBS_IDX)
    return out


# ---------------------------------------------------------------------------
# Parameter estimation (lambda = (A, B, pi)) via counting + Laplace smoothing
# ---------------------------------------------------------------------------
def estimate_params(train_df: pd.DataFrame, alpha: float = 1.0):
    states_seq = train_df["true_state"].map(STATE_IDX).to_numpy()
    obs_seq = train_df["obs_idx"].to_numpy()

    # --- pi: initial state distribution ---
    # Single continuous sequence here -> pi estimated from overall state
    # frequency (a reasonable choice when there's one long trace rather than
    # many independent short sequences). Document this choice in the report.
    pi_counts = np.bincount(states_seq, minlength=N_STATES).astype(float)
    pi = (pi_counts + alpha) / (pi_counts.sum() + alpha * N_STATES)

    # --- A: transition matrix, counted from consecutive true states ---
    A_counts = np.zeros((N_STATES, N_STATES))
    for t in range(len(states_seq) - 1):
        A_counts[states_seq[t], states_seq[t + 1]] += 1
    A = (A_counts + alpha) / (A_counts.sum(axis=1, keepdims=True) + alpha * N_STATES)

    # --- B: emission table, P(obs_symbol | state) ---
    B_counts = np.zeros((N_STATES, N_OBS))
    for s, o in zip(states_seq, obs_seq):
        B_counts[s, o] += 1
    B = (B_counts + alpha) / (B_counts.sum(axis=1, keepdims=True) + alpha * N_OBS)

    return A, B, pi


# ---------------------------------------------------------------------------
# Forward algorithm (log-space to avoid underflow)
# ---------------------------------------------------------------------------
def forward_log(obs_seq, A, B, pi):
    T = len(obs_seq)
    n_states = len(pi)
    log_A = np.log(A)
    log_B = np.log(B)
    log_pi = np.log(pi)

    log_alpha = np.zeros((T, n_states))
    log_alpha[0] = log_pi + log_B[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(n_states):
            log_alpha[t, j] = (
                logsumexp(log_alpha[t - 1] + log_A[:, j]) + log_B[j, obs_seq[t]]
            )

    log_likelihood = logsumexp(log_alpha[-1])
    return log_alpha, log_likelihood


def logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


# ---------------------------------------------------------------------------
# Viterbi algorithm (log-space)
# ---------------------------------------------------------------------------
def viterbi(obs_seq, A, B, pi):
    T = len(obs_seq)
    n_states = len(pi)
    log_A = np.log(A)
    log_B = np.log(B)
    log_pi = np.log(pi)

    delta = np.zeros((T, n_states))
    psi = np.zeros((T, n_states), dtype=int)

    delta[0] = log_pi + log_B[:, obs_seq[0]]

    for t in range(1, T):
        for j in range(n_states):
            scores = delta[t - 1] + log_A[:, j]
            psi[t, j] = np.argmax(scores)
            delta[t, j] = np.max(scores) + log_B[j, obs_seq[t]]

    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path, delta


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true_idx, y_pred_idx):
    n = len(y_true_idx)
    accuracy = float(np.mean(y_true_idx == y_pred_idx))

    confusion = np.zeros((N_STATES, N_STATES), dtype=int)
    for t, p in zip(y_true_idx, y_pred_idx):
        confusion[t, p] += 1

    # Macro F1 (computed manually, no sklearn dependency required)
    f1_scores = []
    for s in range(N_STATES):
        tp = confusion[s, s]
        fp = confusion[:, s].sum() - tp
        fn = confusion[s, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        f1_scores.append(f1)
    macro_f1 = float(np.mean(f1_scores))

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": confusion,
        "per_class_f1": dict(zip(STATES, f1_scores)),
        "n": n,
    }


def print_confusion(confusion):
    print("\nConfusion matrix (rows=true, cols=predicted):")
    header = "        " + "  ".join(f"{s:>5}" for s in STATES)
    print(header)
    for i, s in enumerate(STATES):
        row = "  ".join(f"{confusion[i, j]:>5}" for j in range(N_STATES))
        print(f"{s:>6}  {row}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to clean_data.csv (from preprocess.py)")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--alpha", type=float, default=1.0, help="Laplace smoothing constant")
    ap.add_argument("--output-dir", default="../output")
    args = ap.parse_args()

    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    print(f"Loaded {len(df)} rows.")

    # Chronological split -- NO shuffling
    split_idx = int(len(df) * args.train_frac)
    train_df, test_df = df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows "
          f"(split at {args.train_frac:.0%}, chronological)")

    # Discretize (thresholds fit on TRAIN only)
    thresholds = fit_thresholds(train_df)
    print("\nDiscretization thresholds (fit on TRAIN only):")
    for k, v in thresholds.items():
        print(f"  {k}: > {v:.2f} -> High/Motion, else Low/No-motion")

    train_df = discretize(train_df, thresholds)
    test_df = discretize(test_df, thresholds)

    # Estimate HMM parameters from TRAIN
    A, B, pi = estimate_params(train_df, alpha=args.alpha)

    print("\nTransition matrix A (rows sum to 1):")
    print(pd.DataFrame(A, index=STATES, columns=STATES).round(4))
    print("\nInitial distribution pi:")
    print(pd.Series(pi, index=STATES).round(4))
    assert np.allclose(A.sum(axis=1), 1.0), "A rows must sum to 1"
    assert np.allclose(B.sum(axis=1), 1.0), "B rows must sum to 1"
    assert np.isclose(pi.sum(), 1.0), "pi must sum to 1"
    print("\n[OK] A, B, pi are valid probability distributions.")

    # --- Run Forward + Viterbi on TEST ---
    test_obs = test_df["obs_idx"].to_numpy()
    test_true_idx = test_df["true_state"].map(STATE_IDX).to_numpy()

    t0 = time.time()
    path, _ = viterbi(test_obs, A, B, pi)
    viterbi_time = time.time() - t0

    t0 = time.time()
    _, log_likelihood = forward_log(test_obs, A, B, pi)
    forward_time = time.time() - t0

    # --- Evaluate ---
    results = evaluate(test_true_idx, path)
    print("\n" + "=" * 60)
    print("TEST SET RESULTS (Viterbi-decoded state sequence)")
    print("=" * 60)
    print(f"Accuracy          : {results['accuracy']:.4f}")
    print(f"Macro F1          : {results['macro_f1']:.4f}")
    print(f"Per-class F1      : {results['per_class_f1']}")
    print(f"Log-likelihood    : {log_likelihood:.2f}  (Forward algorithm, test sequence)")
    print(f"Viterbi time      : {viterbi_time*1000:.2f} ms  ({len(test_obs)} steps)")
    print(f"Forward time      : {forward_time*1000:.2f} ms  ({len(test_obs)} steps)")
    print_confusion(results["confusion_matrix"])

    # --- Export predictions in the team's agreed hand-off format ---
    pred_states = [STATES[i] for i in path]
    # Per-step state probabilities via normalized forward-backward would be
    # more correct; as a documented approximation we report a one-hot-ish
    # confidence from the Viterbi path's local delta margin isn't exported
    # here -- instead we export Forward-filtered P(state_t | obs_1..t),
    # which is the standard "current belief" signal for Member 3.
    log_alpha, _ = forward_log(test_obs, A, B, pi)
    alpha = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    alpha = alpha / alpha.sum(axis=1, keepdims=True)  # normalize per row

    out_df = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "predicted_state": pred_states,
        "P(S0)": alpha[:, 0],
        "P(S1)": alpha[:, 1],
        "P(S2)": alpha[:, 2],
        "true_state": test_df["true_state"].values,
    })
    out_path = f"{args.output_dir}/hmm_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote predictions (hand-off format for Member 3) to {out_path}")

    # Save metrics summary too
    metrics_path = f"{args.output_dir}/hmm_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Per-class F1: {results['per_class_f1']}\n")
        f.write(f"Log-likelihood (test, Forward): {log_likelihood:.2f}\n")
        f.write(f"Viterbi inference time (s): {viterbi_time:.4f}\n")
        f.write(f"Forward inference time (s): {forward_time:.4f}\n")
        f.write(f"Test set size: {len(test_obs)}\n")
        f.write("\nConfusion matrix (rows=true, cols=predicted):\n")
        f.write(str(pd.DataFrame(results["confusion_matrix"], index=STATES, columns=STATES)))
    print(f"Wrote metrics summary to {metrics_path}")


if __name__ == "__main__":
    main()
