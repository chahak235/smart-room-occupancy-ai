"""
Member 1 — Preprocessing for the Smart Room HMM.

Loads the UCI Room Occupancy Estimation dataset (Occupancy_Estimation.csv),
combines Date+Time into a sortable timestamp, maps the raw Room_Occupancy_Count
(0-3 people) to the project's 3-state occupancy label (S0/S1/S2), discretizes
the 4 sensor families used by the HMM (CO2, PIR, Sound, Light) into 2 bins
each, and writes out a clean, time-ordered CSV ready for the HMM script.

Expected input columns (UCI dataset 864, 10,129 rows, 18 features):
Date, Time, S1_Temp, S2_Temp, S3_Temp, S4_Temp, S1_Light, S2_Light,
S3_Light, S4_Light, S1_Sound, S2_Sound, S3_Sound, S4_Sound, S5_CO2,
S5_CO2_Slope, S6_PIR, S7_PIR, Room_Occupancy_Count

Usage:
    python preprocess.py --input /path/to/Occupancy_Estimation.csv \
                          --output ../output/clean_data.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# State mapping: Room_Occupancy_Count (0-3 people) -> S0/S1/S2
# ---------------------------------------------------------------------------
def map_occupancy_to_state(count: int) -> str:
    if count == 0:
        return "S0"       # Empty
    elif count == 1:
        return "S1"       # Low occupancy
    else:                 # 2 or 3 people
        return "S2"       # Higher occupancy


# ---------------------------------------------------------------------------
# Sensor aggregation: the dataset has 4 nodes for Temp/Light/Sound (S1-S4)
# and single CO2 (S5) + two PIR sensors (S6, S7). We aggregate the
# multi-node readings into one signal per sensor family, matching the
# project's 4-sensor observation model (CO2, PIR, Sound, Light).
# ---------------------------------------------------------------------------
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
    # Motion = 1 if either PIR sensor fired
    out["pir"] = df[pir_cols].max(axis=1)

    return out


# ---------------------------------------------------------------------------
# Discretization: split CO2/Light/Sound into Low/High at the training-set
# median (PIR is already binary: 0 = no motion, 1 = motion).
# IMPORTANT: thresholds must be computed on the TRAIN split only, then
# applied to test, to avoid leakage. Since this script runs before the
# train/test split, it reports the full-data median here for inspection;
# hmm_model.py recomputes thresholds from the train split only.
# ---------------------------------------------------------------------------
def report_thresholds(agg: pd.DataFrame):
    print("\n--- Sensor summary (full dataset, for sanity-checking only) ---")
    for col in ["co2", "light", "sound"]:
        print(f"{col:>6}: min={agg[col].min():.2f}  median={agg[col].median():.2f}  "
              f"max={agg[col].max():.2f}")
    motion_rate = agg["pir"].mean()
    print(f"  pir : motion-rate={motion_rate:.3f} (fraction of rows with motion=1)")
    print("NOTE: actual Low/High thresholds used by the HMM/Bayes models are "
          "computed on the TRAIN split only (see hmm_model.py), not here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to raw Occupancy_Estimation.csv")
    ap.add_argument("--output", default="../output/clean_data.csv", help="Path to write cleaned CSV")
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
    print("Columns found:", list(df.columns))

    if "Date" not in df.columns or "Time" not in df.columns:
        sys.exit("ERROR: expected 'Date' and 'Time' columns for timestamp construction.")

    # Build a proper sortable timestamp and sort chronologically
    df["timestamp"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Check for duplicate/out-of-order timestamps
    n_dupes = df["timestamp"].duplicated().sum()
    if n_dupes:
        print(f"WARNING: {n_dupes} duplicate timestamps found (kept as-is, order preserved).")

    if "Room_Occupancy_Count" not in df.columns:
        sys.exit("ERROR: expected target column 'Room_Occupancy_Count' not found.")

    # Map target to 3-state label
    df["true_state"] = df["Room_Occupancy_Count"].astype(int).apply(map_occupancy_to_state)

    # Aggregate sensors
    agg = aggregate_sensors(df)
    report_thresholds(agg)

    clean = pd.concat(
        [df[["timestamp", "Room_Occupancy_Count", "true_state"]], agg],
        axis=1,
    )

    print("\nState distribution (full dataset):")
    print(clean["true_state"].value_counts().sort_index())

    clean.to_csv(args.output, index=False)
    print(f"\nWrote cleaned, time-ordered data to {args.output} ({len(clean)} rows).")
    print("Columns: timestamp, Room_Occupancy_Count, true_state, co2, light, sound, pir")


if __name__ == "__main__":
    main()
