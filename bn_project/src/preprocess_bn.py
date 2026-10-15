"""
Member 2 — Preprocessing for the Smart Room Bayesian Network.

Loads the UCI Room Occupancy Estimation dataset (Occupancy_Estimation.csv),
maps the raw Room_Occupancy_Count (0-3 people) to the project's 3-state
occupancy label (S0/S1/S2), and aggregates the 4 sensor families used by
the Bayesian Network (CO2, PIR, Sound, Light).

Unlike Member 1's HMM preprocessing, this script does NOT need to preserve
row order or build a timestamp for sequence modelling — the Bayesian
Network treats every row as an independent, single-instant snapshot
(the whole point of this technique is "estimate occupancy from ONE
reading", with no memory of the previous reading). The timestamp is kept
only as a reference column, not as something the model conditions on.

Expected input columns (UCI dataset 864, 10,129 rows, 18 features):
Date, Time, S1_Temp, S2_Temp, S3_Temp, S4_Temp, S1_Light, S2_Light,
S3_Light, S4_Light, S1_Sound, S2_Sound, S3_Sound, S4_Sound, S5_CO2,
S5_CO2_Slope, S6_PIR, S7_PIR, Room_Occupancy_Count

Usage:
    python preprocess_bn.py --input /path/to/Occupancy_Estimation.csv \
                             --output ../output/clean_data_bn.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# State mapping: Room_Occupancy_Count (0-3 people) -> S0/S1/S2
# (identical mapping to Member 1's HMM, so both techniques' outputs line up
# with the same ground-truth labels and with each other)
# ---------------------------------------------------------------------------
def map_occupancy_to_state(count: int) -> str:
    if count == 0:
        return "S0"       # Empty
    elif count == 1:
        return "S1"       # Low occupancy
    else:                 # 2 or 3 people
        return "S2"       # Higher occupancy


def aggregate_sensors(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    light_cols = [c for c in ["S1_Light", "S2_Light", "S3_Light", "S4_Light"] if c in df.columns]
    sound_cols = [c for c in ["S1_Sound", "S2_Sound", "S3_Sound", "S4_Sound"] if c in df.columns]
    pir_cols = [c for c in ["S6_PIR", "S7_PIR"] if c in df.columns]

    missing = []
    if not light_cols:
        missing.append("Light (S1_Light..S4_Light)")
    if not sound_cols:
        missing.append("Sound (S1_Sound..S4_Sound)")
    if "S5_CO2" not in df.columns:
        missing.append("CO2 (S5_CO2)")
    if not pir_cols:
        missing.append("PIR (S6_PIR, S7_PIR)")
    if missing:
        raise ValueError(
            "Input CSV is missing expected columns: " + ", ".join(missing) +
            ". Check that you're using the raw UCI Occupancy_Estimation.csv "
            "(18 features) and haven't renamed any columns."
        )

    out["co2"] = df["S5_CO2"]
    out["light"] = df[light_cols].mean(axis=1)
    out["sound"] = df[sound_cols].mean(axis=1)
    out["pir"] = df[pir_cols].max(axis=1)   # motion = 1 if either PIR fired

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw Occupancy_Estimation.csv")
    ap.add_argument("--output", default="../output/clean_data_bn.csv", help="Path to write cleaned CSV")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

    if "Date" in df.columns and "Time" in df.columns:
        df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    else:
        df["timestamp"] = pd.RangeIndex(len(df))

    if "Room_Occupancy_Count" not in df.columns:
        sys.exit("ERROR: expected target column 'Room_Occupancy_Count' not found.")

    df["true_state"] = df["Room_Occupancy_Count"].astype(int).apply(map_occupancy_to_state)

    agg = aggregate_sensors(df)

    clean = pd.concat(
        [df[["timestamp", "Room_Occupancy_Count", "true_state"]], agg],
        axis=1,
    )

    print("\nState distribution (full dataset):")
    print(clean["true_state"].value_counts().sort_index())

    print("\n--- Sensor summary (full dataset, for sanity-checking only) ---")
    for col in ["co2", "light", "sound"]:
        print(f"{col:>6}: min={agg[col].min():.2f}  median={agg[col].median():.2f}  max={agg[col].max():.2f}")
    print(f"  pir : motion-rate={agg['pir'].mean():.3f}")
    print("NOTE: actual discretization thresholds used by the BN are computed "
          "on the TRAIN split only (see bn_model.py), not here — this is a "
          "full-dataset summary for sanity-checking, not the model's thresholds.")

    clean.to_csv(args.output, index=False)
    print(f"\nWrote cleaned data to {args.output} ({len(clean)} rows).")
    print("Columns: timestamp, Room_Occupancy_Count, true_state, co2, light, sound, pir")
    print("\nNote: rows are kept in original chronological order in the file for "
          "traceability, but bn_model.py treats them as i.i.d. snapshots and "
          "splits train/test with a RANDOM stratified shuffle, not chronologically.")


if __name__ == "__main__":
    main()
