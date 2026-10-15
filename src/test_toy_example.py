"""
Member 1 — Sanity check for the Forward and Viterbi implementations.

Runs our hmm_model.forward_log / hmm_model.viterbi functions on a tiny,
hand-verifiable 2-state HMM (the classic "Rainy/Sunny -> walk/shop/clean"
example from Rabiner's tutorial, reduced to 2 states and 3 observations)
and checks the output against the known-correct answer worked out by hand.

If this script passes, you have real confidence in the Forward/Viterbi
code before trusting it on the full 3-state, 16-symbol room dataset.

Run:
    python test_toy_example.py
"""

import numpy as np
from hmm_model import forward_log, viterbi, logsumexp

# --- Toy 2-state HMM (Rabiner-style weather example) ---
# States: 0 = Rainy, 1 = Sunny
# Observations: 0 = walk, 1 = shop, 2 = clean
TOY_STATES = ["Rainy", "Sunny"]
TOY_OBS = ["walk", "shop", "clean"]

pi = np.array([0.6, 0.4])

A = np.array([
    [0.7, 0.3],   # Rainy -> Rainy, Rainy -> Sunny
    [0.4, 0.6],   # Sunny -> Rainy, Sunny -> Sunny
])

B = np.array([
    [0.1, 0.4, 0.5],   # Rainy: P(walk), P(shop), P(clean)
    [0.6, 0.3, 0.1],   # Sunny: P(walk), P(shop), P(clean)
])

# Observation sequence: walk, shop, clean  (classic textbook example)
obs_seq = [0, 1, 2]


def hand_worked_viterbi():
    """
    Hand-worked Viterbi trellis for this exact example. This is the
    classic Rainy/Sunny -> walk/shop/clean example (same numbers as the
    worked example on Wikipedia's "Viterbi algorithm" page), used here
    purely as an independently-checkable ground truth.

    t=0 (walk):
      delta_Rainy(0) = pi(Rainy)*B(Rainy,walk)   = 0.6 * 0.1 = 0.06
      delta_Sunny(0) = pi(Sunny)*B(Sunny,walk)   = 0.4 * 0.6 = 0.24

    t=1 (shop):
      delta_Rainy(1) = max(0.06*0.7, 0.24*0.4) * B(Rainy,shop)
                      = max(0.042, 0.096) * 0.4 = 0.096*0.4 = 0.0384  (best predecessor: Sunny)
      delta_Sunny(1) = max(0.06*0.3, 0.24*0.6) * B(Sunny,shop)
                      = max(0.018, 0.144) * 0.3 = 0.144*0.3 = 0.0432  (best predecessor: Sunny)

    t=2 (clean):
      delta_Rainy(2) = max(delta_Rainy(1)*0.7, delta_Sunny(1)*0.4) * B(Rainy,clean)
                      = max(0.0384*0.7, 0.0432*0.4) * 0.5
                      = max(0.02688, 0.01728) * 0.5 = 0.02688*0.5 = 0.01344  (best predecessor: Rainy)
      delta_Sunny(2) = max(delta_Rainy(1)*0.3, delta_Sunny(1)*0.6) * B(Sunny,clean)
                      = max(0.0384*0.3, 0.0432*0.6) * 0.1
                      = max(0.01152, 0.02592) * 0.1 = 0.02592*0.1 = 0.002592 (best predecessor: Sunny)

    Best final state at t=2: Rainy (0.01344 > 0.002592).

    Backtrack (this is where it's easy to go wrong by hand -- follow the
    "best predecessor" recorded at EACH step, not the raw delta values):
      path[2] = Rainy
      path[1] = best predecessor of Rainy at t=2        = Rainy   (0.02688 branch, not 0.01728)
      path[0] = best predecessor of Rainy at t=1         = Sunny   (0.096 branch, not 0.042)

    Expected best path: [Sunny, Rainy, Rainy], probability 0.01344.
    """
    return ["Sunny", "Rainy", "Rainy"]


def main():
    print("=== Sanity check: Forward + Viterbi on a hand-worked toy HMM ===\n")

    expected_path = hand_worked_viterbi()
    print("Hand-worked expected path:", expected_path)

    path_idx, delta = viterbi(obs_seq, A, B, pi)
    predicted_path = [TOY_STATES[i] for i in path_idx]
    print("Code-computed path       :", predicted_path)

    assert predicted_path == expected_path, (
        f"MISMATCH: expected {expected_path}, got {predicted_path}. "
        "Check the Viterbi implementation before trusting it on real data."
    )
    print("[PASS] Viterbi matches the hand-worked solution.\n")

    # Forward algorithm: check P(obs_seq) by brute force over all 2^3 = 8
    # possible state sequences, then compare to the Forward algorithm's
    # log-likelihood.
    log_alpha, log_likelihood = forward_log(obs_seq, A, B, pi)
    print(f"Forward algorithm log-likelihood: {log_likelihood:.6f}")
    print(f"Forward algorithm likelihood    : {np.exp(log_likelihood):.6f}")

    brute_force_total = 0.0
    import itertools
    for seq in itertools.product([0, 1], repeat=3):
        p = pi[seq[0]] * B[seq[0], obs_seq[0]]
        for t in range(1, 3):
            p *= A[seq[t - 1], seq[t]] * B[seq[t], obs_seq[t]]
        brute_force_total += p

    print(f"Brute-force likelihood (sum over all state paths): {brute_force_total:.6f}")
    assert np.isclose(np.exp(log_likelihood), brute_force_total, atol=1e-8), (
        "MISMATCH: Forward algorithm result doesn't match brute-force enumeration."
    )
    print("[PASS] Forward algorithm matches brute-force enumeration.\n")

    print("=== All sanity checks passed. Safe to run on the real dataset. ===")


if __name__ == "__main__":
    main()
