"""
Member 1 — Visualizations for the HMM occupancy-estimation results.

Produces, from the actual run artifacts (clean_data.csv + hmm_predictions.csv):
  1. confusion_matrix.png      - heatmap of true vs predicted state
  2. transition_matrix.png     - heatmap of the learned A matrix
  3. occupancy_timeline.png    - true vs predicted state over the test window
  4. state_distribution.png    - train vs test class balance (explains the S1 F1=0 finding)
  5. state_probabilities.png   - HMM's P(S0)/P(S1)/P(S2) belief over time (first 300 test steps)

Run (from the same folder as clean_data.csv and hmm_predictions.csv):
    python make_visuals.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STATES = ["S0", "S1", "S2"]
STATE_LABELS = {"S0": "Empty", "S1": "Low (1 person)", "S2": "Higher (2-3 people)"}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})


def plot_confusion_matrix(pred_df, out_path):
    cm = pd.crosstab(pred_df["true_state"], pred_df["predicted_state"]) \
            .reindex(index=STATES, columns=STATES, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm.values, cmap="Blues")

    ax.set_xticks(range(3)); ax.set_xticklabels([STATE_LABELS[s] for s in STATES], rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([STATE_LABELS[s] for s in STATES])
    ax.set_xlabel("Predicted state (Viterbi)")
    ax.set_ylabel("True state")
    ax.set_title("HMM Confusion Matrix — Test Set")

    vmax = cm.values.max()
    for i in range(3):
        for j in range(3):
            val = cm.values[i, j]
            color = "white" if val > vmax * 0.5 else "black"
            ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_transition_matrix(A, out_path):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(A, cmap="Purples", vmin=0, vmax=1)
    ax.set_xticks(range(3)); ax.set_xticklabels(STATES)
    ax.set_yticks(range(3)); ax.set_yticklabels(STATES)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")
    ax.set_title("Learned Transition Matrix A\n(estimated from training data)")

    for i in range(3):
        for j in range(3):
            val = A[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", color=color, fontsize=11)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(next | current)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_state_distribution(clean_df, split_idx, out_path):
    train = clean_df.iloc[:split_idx]
    test = clean_df.iloc[split_idx:]

    train_counts = train["true_state"].value_counts().reindex(STATES, fill_value=0)
    test_counts = test["true_state"].value_counts().reindex(STATES, fill_value=0)

    x = np.arange(3)
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    b1 = ax.bar(x - width/2, train_counts.values, width, label="Train", color="#4C72B0")
    b2 = ax.bar(x + width/2, test_counts.values, width, label="Test", color="#DD8452")

    ax.set_xticks(x); ax.set_xticklabels([STATE_LABELS[s] for s in STATES])
    ax.set_ylabel("Row count")
    ax.set_title("Class Balance: Train vs Test (chronological split)")
    ax.legend()
    ax.bar_label(b1, padding=2, fontsize=9)
    ax.bar_label(b2, padding=2, fontsize=9)

    # Annotate the S1 finding directly on the chart
    ax.annotate("Test set has 0 S1 rows ->\nS1 F1-score is undefined/0\n(not a bug: a real effect of\nchronological splitting on\nan imbalanced, non-stationary dataset)",
                xy=(1 + width/2, 0), xytext=(1.15, train_counts.max()*0.55),
                fontsize=8.5, color="#B03A2E",
                arrowprops=dict(arrowstyle="->", color="#B03A2E"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_timeline(pred_df, out_path, n=600):
    sub = pred_df.iloc[:n].copy()
    state_to_num = {"S0": 0, "S1": 1, "S2": 2}
    true_num = sub["true_state"].map(state_to_num)
    pred_num = sub["predicted_state"].map(state_to_num)

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(range(len(sub)), true_num, where="post", label="True state", color="#2E7D32", linewidth=2)
    ax.step(range(len(sub)), pred_num, where="post", label="HMM predicted (Viterbi)",
            color="#C62828", linewidth=1.2, linestyle="--", alpha=0.85)

    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["S0 Empty", "S1 Low", "S2 Higher"])
    ax.set_xlabel(f"Test-set time step (first {n} of {len(pred_df)}, ~30s apart)")
    ax.set_title("True vs. HMM-Predicted Occupancy State Over Time")
    ax.legend(loc="upper right")
    ax.set_ylim(-0.3, 2.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_belief_probabilities(pred_df, out_path, n=300):
    sub = pred_df.iloc[:n]
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.stackplot(
        range(len(sub)),
        sub["P(S0)"], sub["P(S1)"], sub["P(S2)"],
        labels=["P(S0) Empty", "P(S1) Low", "P(S2) Higher"],
        colors=["#90CAF9", "#FFCC80", "#EF9A9A"],
        alpha=0.9,
    )
    ax.set_xlabel(f"Test-set time step (first {n})")
    ax.set_ylabel("Forward-filtered belief")
    ax.set_title("HMM Belief P(state | observations so far) Over Time")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    import sys
    sys.path.insert(0, ".")
    from hmm_model import fit_thresholds, discretize, estimate_params, STATE_IDX

    clean_df = pd.read_csv("clean_data.csv", parse_dates=["timestamp"])
    pred_df = pd.read_csv("hmm_predictions.csv", parse_dates=["timestamp"])

    split_idx = int(len(clean_df) * 0.8)
    train_df = clean_df.iloc[:split_idx].copy()
    thresholds = fit_thresholds(train_df)
    train_df = discretize(train_df, thresholds)
    A, B, pi = estimate_params(train_df)

    plot_confusion_matrix(pred_df, "confusion_matrix.png")
    plot_transition_matrix(A, "transition_matrix.png")
    plot_state_distribution(clean_df, split_idx, "state_distribution.png")
    plot_timeline(pred_df, "occupancy_timeline.png")
    plot_belief_probabilities(pred_df, "state_probabilities.png")

    print("\nAll visuals written to the current folder.")


if __name__ == "__main__":
    main()