"""
Member 2 — Visualizations for the Bayesian Network occupancy-estimation results.

Produces, from the actual run artifacts (bn_params.json + bn_predictions.csv):
  1. network_structure.png   - the BN's graph structure (Occupancy -> 4 sensors)
  2. confusion_matrix_bn.png - heatmap of true vs predicted state
  3. cpt_heatmaps.png        - P(sensor=High | Occupancy) for all 4 sensors
  4. prior_distribution.png  - learned P(Occupancy) prior vs actual class balance
  5. posterior_examples.png  - example posteriors for a few test rows

Run (from the same folder as bn_params.json and bn_predictions.csv):
    python make_visuals_bn.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STATES = ["S0", "S1", "S2"]
STATE_LABELS = {"S0": "Empty", "S1": "Low (1 person)", "S2": "Higher (2-3 people)"}
SENSORS = ["co2", "light", "sound", "pir"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})


def plot_network_structure(out_path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Bayesian Network Structure\n(Occupancy causes conditionally-independent sensor readings)")

    root_pos = (5.5, 8)
    child_positions = {
        "CO2": (1.5, 3),
        "Light": (4.3, 3),
        "Sound": (7.1, 3),
        "PIR": (9.9, 3),
    }

    root = mpatches.FancyBboxPatch((root_pos[0] - 1.3, root_pos[1] - 0.6), 2.6, 1.2,
                                    boxstyle="round,pad=0.1", facecolor="#4C72B0", edgecolor="black")
    ax.add_patch(root)
    ax.text(root_pos[0], root_pos[1], "Occupancy\n(S0 / S1 / S2)", ha="center", va="center",
            color="white", fontweight="bold", fontsize=11)

    for label, pos in child_positions.items():
        box = mpatches.FancyBboxPatch((pos[0] - 1.0, pos[1] - 0.55), 2.0, 1.1,
                                       boxstyle="round,pad=0.08", facecolor="#DD8452", edgecolor="black")
        ax.add_patch(box)
        ax.text(pos[0], pos[1], label, ha="center", va="center", fontweight="bold", fontsize=10)
        ax.annotate("", xy=(pos[0], pos[1] + 0.6), xytext=(root_pos[0], root_pos[1] - 0.65),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                     connectionstyle="arc3,rad=0.0"))

    ax.text(5.5, 1.0,
            "Given the true Occupancy state, CO2 / Light / Sound / PIR are modelled as\n"
            "conditionally independent of each other (the \"naive Bayes\" network shape).\n"
            "Inference: P(Occupancy | all 4 readings)  ∝  P(Occupancy) · Π P(sensor | Occupancy)",
            ha="center", va="center", fontsize=9.5, style="italic", color="#333333")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_confusion_matrix(pred_df, out_path):
    cm = pd.crosstab(pred_df["true_state"], pred_df["predicted_state"]) \
            .reindex(index=STATES, columns=STATES, fill_value=0)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm.values, cmap="Greens")

    ax.set_xticks(range(3)); ax.set_xticklabels([STATE_LABELS[s] for s in STATES], rotation=20, ha="right")
    ax.set_yticks(range(3)); ax.set_yticklabels([STATE_LABELS[s] for s in STATES])
    ax.set_xlabel("Predicted state (MAP posterior)")
    ax.set_ylabel("True state")
    ax.set_title("Bayesian Network Confusion Matrix — Test Set")

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


def plot_cpt_heatmaps(cpts, out_path):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    for ax, sensor in zip(axes, SENSORS):
        mat = np.array([[cpts[sensor][s]["Low"], cpts[sensor][s]["High"]] for s in STATES])
        im = ax.imshow(mat, cmap="Oranges", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Low", "High"])
        ax.set_yticks(range(3)); ax.set_yticklabels(STATES)
        ax.set_title(f"P({sensor} | Occupancy)")
        for i in range(3):
            for j in range(2):
                val = mat[i, j]
                color = "white" if val > 0.5 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
    fig.suptitle("Learned Conditional Probability Tables (CPTs)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_prior_distribution(priors, pred_df, out_path):
    actual_counts = pred_df["true_state"].value_counts().reindex(STATES, fill_value=0)
    actual_frac = actual_counts / actual_counts.sum()
    prior_frac = pd.Series(priors).reindex(STATES)

    x = np.arange(3)
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    b1 = ax.bar(x - width/2, prior_frac.values, width, label="Learned prior P(Occupancy)", color="#4C72B0")
    b2 = ax.bar(x + width/2, actual_frac.values, width, label="Actual test-set frequency", color="#55A868")
    ax.set_xticks(x); ax.set_xticklabels([STATE_LABELS[s] for s in STATES])
    ax.set_ylabel("Probability / fraction")
    ax.set_title("Learned Prior vs. Actual Class Balance\n(random stratified split -- every class appears in test)")
    ax.legend()
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=9)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_posterior_examples(pred_df, out_path, n=12):
    sub = pred_df.sample(n=min(n, len(pred_df)), random_state=7).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(sub))
    colors = {"P(S0)": "#90CAF9", "P(S1)": "#FFCC80", "P(S2)": "#EF9A9A"}
    for col in ["P(S0)", "P(S1)", "P(S2)"]:
        ax.bar(range(len(sub)), sub[col], bottom=bottom, label=col, color=colors[col])
        bottom += sub[col].values

    labels = [f"true={row.true_state}\npred={row.predicted_state}" for _, row in sub.iterrows()]
    ax.set_xticks(range(len(sub))); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Posterior probability")
    ax.set_title(f"Example Posteriors P(Occupancy | sensor readings) for {len(sub)} Random Test Rows")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 1.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    with open("../output/bn_params.json") as f:
        params = json.load(f)
    pred_df = pd.read_csv("../output/bn_predictions.csv", parse_dates=["timestamp"])

    plot_network_structure("../output/network_structure.png")
    plot_confusion_matrix(pred_df, "../output/confusion_matrix_bn.png")
    plot_cpt_heatmaps(params["cpts"], "../output/cpt_heatmaps.png")
    plot_prior_distribution(params["priors"], pred_df, "../output/prior_distribution.png")
    plot_posterior_examples(pred_df, "../output/posterior_examples.png")

    print("\nAll visuals written to ../output/")


if __name__ == "__main__":
    main()
