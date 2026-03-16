#!/usr/bin/env python3
"""
weta_temperature_mapping.py
===========================
Step 2: Combine already-processed mm trajectories (from weta_trajectory_processor.py)
with per-frame temperature sensor data.

Reads:
  - {trial_id}_center_mm.npy  from processed_trajectories/
  - Temperature .txt files    from temperature_experiments/

Writes:
  - {trial_id}_animal_temperature.npy   (N,) interpolated °C at animal x-pos
  - {trial_id}_temp_sensors.npy         (N,10) raw sensor readings
  - temperature_summary.json

Temperature setup
-----------------
10 sensors linearly spaced along the 600 mm arena x-axis.
  Column 0 = 600 mm (hot end, right)
  Column 9 =   0 mm (cold end, left)

Usage
-----
    python weta_temperature_mapping.py \
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped
"""

import numpy as np
import json
import os
import argparse

# ──────────────────────────────────────────────────────────────────────────────
# Explicit trial_id → temperature filename mapping
# ──────────────────────────────────────────────────────────────────────────────
TRIAL_TO_TEMP = {
    # H. crassidens
    "hcrass1f_trial_redo_20250203_120147": "Hcrass1f trial redo.txt",
    "hcrass2f_trial_20250115_135415":      "Hcrass2f trial.txt",
    "hcrass2f_trial_redo_20250203_131724": "Hcrass2f trial redo.txt",
    "hcrass3f_trial_20250205_125459":      "Hcrass3f trial redo.txt",
    "hcrass3_trial_20250117_110858":       "Hcrass3trial.txt",
    "hcrass4m_trial_20250205_113922":      "Hcrass4m trial.txt",
    "hcrass5f_trial_20250122_122502":      "Hcrass5F trial.txt",
    "hcrass6f_trial_20250123_112349":      "Hcrass6F trial.txt",
    "hcrass7f_trial_20250127_115538":      "Hcrass7f trial.txt",
    "hcrass8m_trial_20250127_131249":      "Hcrass8m trial.txt",
    # H. maori
    "hm_10_trial_20250310_110658":         "HM10 trial.txt",
    "hm11_trial_20250309_120332":          "HM11 trial.txt",
    "hm_12_trial_20250313_123012":         "HM12 trial.txt",
    "hm_15_trial_20250310_153126":         "HM15 trial.txt",
    "hm_16_trial_20250311_100546":         "HM16 trial.txt",
    "hm_18_trial_20250311_113659":         "HM18 trial.txt",
    "hm_19_trial_20250311_134139":         "HM19 trial.txt",
    "hm1_trial_20250309_103617":           "HM1 trial.txt",
    "hm20_trial_20250312_152405":          "HM20 trial.txt",
    "hm22_trial_20250312_135243":          "HM22 trial.txt",
    "hm_23_trial_20250313_153227":         "HM23 trial.txt",
    "hm_2_trial_20250313_140636":          "HM2 trial.txt",
    "hm_3_trial_20250310_140329":          "HM3 trial.txt",
    "hm_4_trial_20250312_121602":          "HM4 trial.txt",
    "hm_8_trial_20250310_123139":          "HM8 trial.txt",
    # H. thoracica
    "hthora_1_trial_20250225_141901":      "Hthora1 trial.txt",
    "hthora_2_trial_20250225_154114":      "Hthora2 trial.txt",
    "hthora_3_trial_20250226_104314":      "Hthora 3 trial.txt",
    "hthora_4_trial_20250226_124247":      "Hthora 4 trial.txt",
    "hthora_5_trial_20250226_140329":      "Hthora 5 trial.txt",
    "hthora_6_trial_20250227_120718":      "Hthora 6 trial.txt",
    "hthora_7_trial_20250227_132041":      "Hthora 7 trial.txt",
    "hthora_8_trial_20250227_143430":      "Hthora 8 trial.txt",
    "hthora_9_trial_20250227_154804":      "Hthora 9 trial.txt",
}

# 10 sensors: index 0 at 600 mm (hot), index 9 at 0 mm (cold)
N_SENSORS = 10
SENSOR_POSITIONS_ASC = [-52, 11, 67, 135, 197, 262, 325, 385, 444, 504]
# ──────────────────────────────────────────────────────────────────────────────
# Temperature I/O and interpolation
# ──────────────────────────────────────────────────────────────────────────────

def _parse_line(line: str):
    """Try to parse a CSV line into a list of floats.
    Returns None if the line contains non-numeric characters (garbled serial)
    or is empty. Returns a (possibly short) list of floats otherwise."""
    line = line.strip().rstrip(",").strip()
    if not line:
        return None
    # Check for letters or other non-numeric junk (allow digits, dots,
    # minus signs, commas, whitespace)
    import re
    if re.search(r"[a-zA-Z]", line):
        return None
    try:
        vals = [float(v.strip()) for v in line.split(",") if v.strip()]
        return vals if len(vals) > 0 else None
    except ValueError:
        return None


def load_temperature_file(filepath: str) -> np.ndarray:
    """Load temperature .txt → (N_lines, 10) array.

    Handles microcontroller serial errors:
    - Lines with letters or unparseable characters → dropped and interpolated.
    - Incomplete lines (wrong number of values)    → dropped and interpolated.
    - Interpolation uses nearest complete (10-value) neighbours.
    - Edge rows filled from nearest complete line.
    """
    # First pass: parse all lines
    raw_rows = []
    with open(filepath, "r") as f:
        for line in f:
            parsed = _parse_line(line)
            raw_rows.append(parsed)  # None for bad lines

    n_rows = len(raw_rows)
    n_expected = N_SENSORS  # 10

    # Build output array, mark bad/incomplete rows as NaN
    out = np.full((n_rows, n_expected), np.nan)
    complete_mask = np.zeros(n_rows, dtype=bool)

    n_garbled = 0
    n_incomplete = 0
    for i, vals in enumerate(raw_rows):
        if vals is None:
            n_garbled += 1
        elif len(vals) == n_expected:
            out[i, :] = vals
            complete_mask[i] = True
        else:
            n_incomplete += 1

    n_bad = n_garbled + n_incomplete
    if n_bad > 0:
        print(f"    [INFO] {os.path.basename(filepath)}: "
              f"{n_garbled} garbled + {n_incomplete} incomplete "
              f"= {n_bad}/{n_rows} bad lines → interpolated")

    # Interpolate bad rows column-by-column using complete neighbours
    complete_idx = np.where(complete_mask)[0]

    if len(complete_idx) == 0:
        raise ValueError(f"No complete lines in {filepath}")

    for col in range(n_expected):
        complete_vals = out[complete_idx, col]
        interpolated = np.interp(np.arange(n_rows), complete_idx, complete_vals)
        out[~complete_mask, col] = interpolated[~complete_mask]

    return out


def resample_temperature(temp_array: np.ndarray, n_target: int) -> np.ndarray:
    """Resample a (N_temp, 10) temperature array to n_target rows.

    Uses linear interpolation along the time axis so the temperature
    timeline is stretched/compressed to match the video frame count.
    """
    n_temp = temp_array.shape[0]
    if n_temp == n_target:
        return temp_array

    src_t = np.linspace(0, 1, n_temp)
    dst_t = np.linspace(0, 1, n_target)
    out = np.zeros((n_target, temp_array.shape[1]))
    for col in range(temp_array.shape[1]):
        out[:, col] = np.interp(dst_t, src_t, temp_array[:, col])
    return out


def interpolate_temperature_at_x(temp_array: np.ndarray,
                                  x_mm: np.ndarray) -> np.ndarray:
    """Interpolate sensor temperatures at the animal's x-position per frame.

    Parameters
    ----------
    temp_array : (N, 10)  — col 0 = 600 mm sensor, col 9 = 0 mm sensor
    x_mm       : (N,)     — animal x-position in mm

    Returns
    -------
    (N,) temperatures in °C
    """
    n_frames = min(len(temp_array), len(x_mm))
    temperatures = np.full(n_frames, np.nan)

    for i in range(n_frames):
        if np.isnan(x_mm[i]):
            continue
        # Flip to ascending x-order for np.interp
        temps_asc = temp_array[i, ::-1]
        temperatures[i] = np.interp(x_mm[i], SENSOR_POSITIONS_ASC, temps_asc)

    return temperatures


# ──────────────────────────────────────────────────────────────────────────────
# Species helper
# ──────────────────────────────────────────────────────────────────────────────

def get_species(trial_id: str) -> str:
    if trial_id.startswith("hcrass"):
        return "H. crassidens"
    elif trial_id.startswith("hm"):
        return "H. maori"
    elif trial_id.startswith("hthora"):
        return "H. thoracica"
    return "unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def run(base_dir: str, processed_dir: str = None):
    temp_dir = os.path.join(base_dir, "temperature_experiments")
    if processed_dir is None:
        processed_dir = os.path.join(base_dir, "processed_trajectories")

    print(f"Processed trajectories : {processed_dir}")
    print(f"Temperature files      : {temp_dir}\n")

    summary = []

    for trial_id, temp_filename in sorted(TRIAL_TO_TEMP.items()):
        center_mm_path = os.path.join(processed_dir, f"{trial_id}_center_mm.npy")
        temp_path = os.path.join(temp_dir, temp_filename)

        # ── Check files exist ──
        if not os.path.isfile(center_mm_path):
            print(f"  [SKIP] {trial_id}: no processed trajectory found")
            continue
        if not os.path.isfile(temp_path):
            print(f"  [SKIP] {trial_id}: temperature file not found → {temp_filename}")
            continue

        # ── Load ──
        center_mm = np.load(center_mm_path)       # (N, 2)
        temp_array = load_temperature_file(temp_path)  # (M, 10)

        n_traj = center_mm.shape[0]
        n_temp = temp_array.shape[0]

        if n_traj != n_temp:
            print(f"    [INFO] Resampling temperature: {n_temp} → {n_traj} frames")
            temp_array = resample_temperature(temp_array, n_traj)

        # ── Interpolate temperature at animal x-position ──
        animal_temp = interpolate_temperature_at_x(temp_array, center_mm[:, 0])

        # ── Save ──
        out_prefix = os.path.join(processed_dir, trial_id)
        np.save(f"{out_prefix}_animal_temperature.npy", animal_temp)
        np.save(f"{out_prefix}_temp_sensors.npy", temp_array)

        valid_t = animal_temp[~np.isnan(animal_temp)]
        species = get_species(trial_id)

        entry = {
            "trial_id": trial_id,
            "species": species,
            "temp_file": temp_filename,
            "n_traj_frames": n_traj,
            "n_temp_frames": n_temp,
        }
        if len(valid_t) > 0:
            entry["temp_min"] = round(float(valid_t.min()), 2)
            entry["temp_max"] = round(float(valid_t.max()), 2)
            entry["temp_mean"] = round(float(valid_t.mean()), 2)

        summary.append(entry)
        print(f"  [OK]   {trial_id:<45} → {temp_filename:<30} "
              f"T: {valid_t.min():.1f}–{valid_t.max():.1f}°C" if len(valid_t) > 0
              else f"  [OK]   {trial_id}")

    # ── Save summary ──
    summary_path = os.path.join(processed_dir, "temperature_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[INFO] Processed {len(summary)} trials.")
    print(f"[INFO] Summary → {summary_path}")
    print(f"[INFO] Per-trial _animal_temperature.npy and _temp_sensors.npy → {processed_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Map processed weta trajectories to temperature data."
    )
    parser.add_argument(
        "--base_dir", type=str,
        default="/home/geuba03p/weta_project/weta_videos_cropped",
    )
    parser.add_argument(
        "--processed_dir", type=str, default=None,
        help="Directory with _center_mm.npy files (default: <base_dir>/processed_trajectories).",
    )
    args = parser.parse_args()
    run(args.base_dir, args.processed_dir)


if __name__ == "__main__":
    main()

# Example usage:  python weta_temperature_mapping.py --base_dir /home/geuba03p/weta_project/weta_videos_cropped
