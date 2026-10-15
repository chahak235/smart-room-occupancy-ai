"""
Member 2 — Sanity check for the Bayesian Network inference function.

Runs our bn_model.posterior() on a tiny, hand-verifiable 2-state network
(classic "Weather -> Umbrella, Mood" naive-Bayes-style Bayesian network)
and checks the output against a posterior worked out by hand with Bayes'
rule.

If this script passes, you have real confidence in the posterior()
function before trusting it on the full 3-state, 4-sensor room dataset.

Run:
    python test_toy_example_bn.py
"""

from bn_model import posterior

# --- Toy 2-state Bayesian network ---
#
#              Weather (Rainy / Sunny)
#             /                      \
#      Umbrella (Yes/No)          Mood (Grumpy/Happy)
#
# Conditionally independent given Weather -- exactly the same "naive
# Bayes" shape as the real Occupancy -> {CO2, Light, Sound, PIR} network,
# just with 2 states and 2 evidence variables instead of 3 states and 4.

TOY_STATES = ["Rainy", "Sunny"]

priors = {"Rainy": 0.5, "Sunny": 0.5}

cpts = {
    "umbrella": {
        "Rainy": {"Yes": 0.9, "No": 0.1},
        "Sunny": {"Yes": 0.2, "No": 0.8},
    },
    "mood": {
        "Rainy": {"Grumpy": 0.7, "Happy": 0.3},
        "Sunny": {"Grumpy": 0.1, "Happy": 0.9},
    },
}

evidence = {"umbrella": "Yes", "mood": "Grumpy"}


def hand_worked_posterior():
    """
    By Bayes' rule, P(state | evidence) ∝ P(state) * Π P(evidence_i | state):

      unnormalized(Rainy) = P(Rainy) * P(Umbrella=Yes|Rainy) * P(Mood=Grumpy|Rainy)
                          = 0.5 * 0.9 * 0.7 = 0.315

      unnormalized(Sunny) = P(Sunny) * P(Umbrella=Yes|Sunny) * P(Mood=Grumpy|Sunny)
                          = 0.5 * 0.2 * 0.1 = 0.01

      total = 0.315 + 0.01 = 0.325

      P(Rainy | evidence) = 0.315 / 0.325 = 0.9692307692...
      P(Sunny | evidence) = 0.01  / 0.325 = 0.0307692308...
    """
    return {"Rainy": 0.315 / 0.325, "Sunny": 0.01 / 0.325}


def main():
    print("=== Sanity check: Bayesian Network posterior on a hand-worked toy example ===\n")

    expected = hand_worked_posterior()
    print("Hand-worked expected posterior:", {k: round(v, 6) for k, v in expected.items()})

    computed = posterior(priors, cpts, evidence, states=TOY_STATES)
    print("Code-computed posterior       :", {k: round(v, 6) for k, v in computed.items()})

    for state in TOY_STATES:
        assert abs(computed[state] - expected[state]) < 1e-8, (
            f"MISMATCH on {state}: expected {expected[state]:.8f}, got {computed[state]:.8f}. "
            "Check the posterior() implementation before trusting it on real data."
        )
    print("[PASS] Computed posterior matches the hand-worked Bayes'-rule solution.\n")

    # Also check it's a valid probability distribution (sums to 1, all >= 0).
    total = sum(computed.values())
    assert abs(total - 1.0) < 1e-9, f"Posterior does not sum to 1 (got {total})."
    assert all(p >= 0 for p in computed.values()), "Posterior has a negative probability."
    print("[PASS] Posterior is a valid probability distribution (sums to 1, non-negative).\n")

    # Sanity-check the qualitative story: umbrella + grumpy mood should push
    # the posterior strongly toward Rainy relative to the 50/50 prior.
    assert computed["Rainy"] > priors["Rainy"], (
        "Evidence pointing toward rain should increase P(Rainy) above the prior."
    )
    print("[PASS] Posterior moves in the right direction relative to the prior.\n")

    print("=== All sanity checks passed. Safe to run on the real dataset. ===")


if __name__ == "__main__":
    main()
