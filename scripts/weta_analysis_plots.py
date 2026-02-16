#!/usr/bin/env python3
"""
weta_analysis_plots.py
======================
Step 3: Analysis and plotting of processed weta trajectory + temperature data.

Reads from processed_trajectories/:
  - {trial_id}_center_mm.npy        (N, 2) x,y in mm
  - {trial_id}_animal_temperature.npy (N,) °C

Produces:
  1. Step histograms  — speed (>2 mm/s) and temperature distributions per species
  2. Boxplots         — median speed and median temperature per animal
  3. Heatmaps         — positional (x, y) and thermal (T, y) per species

Each plot is accompanied by a CSV with the underlying data.

Usage
-----
    python weta_analysis_plots.py \
        --base_dir /home/geuba03p/weta_project/weta_videos_cropped \
        --fps 25
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import re
import csv
import json
import argparse
from collections import defaultdict

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (colourblind-safe, Okabe-Ito)
# ──────────────────────────────────────────────────────────────────────────────
SPECIES_COLOURS = {
    "H. maori":      "#D55E00",
    "H. crassidens": "#56B4E9",
    "H. thoracica":  "#009E73",
}

SPECIES_ORDER = ["H. crassidens", "H. maori", "H. thoracica"]

SPEED_THRESHOLD_MMS = 2.0   # mm/s — only include speeds above this

# ──────────────────────────────────────────────────────────────────────────────
# Trial ID parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_trial_id(trial_id: str) -> dict:
    """Extract species, sex, and animal number from trial_id.

    Examples:
        hcrass1f_trial_redo_20250203_120147  → H. crassidens, F, 1
        hcrass4m_trial_20250205_113922       → H. crassidens, M, 4
        hcrass3_trial_20250117_110858        → H. crassidens, unknown, 3
        hm_10_trial_20250310_110658          → H. maori, unknown, 10
        hm1_trial_20250309_103617            → H. maori, unknown, 1
        hthora_1_trial_20250225_141901       → H. thoracica, unknown, 1
    """
    if trial_id.startswith("hcrass"):
        species = "H. crassidens"
        m = re.match(r"hcrass(\d+)(m|f)?", trial_id)
        if m:
            animal_no = int(m.group(1))
            sex = (m.group(2) or "unknown").upper()
            if sex == "UNKNOWN":
                sex = "unknown"
        else:
            animal_no = 0
            sex = "unknown"

    elif trial_id.startswith("hm"):
        species = "H. maori"
        m = re.match(r"hm_?(\d+)", trial_id)
        animal_no = int(m.group(1)) if m else 0
        sex = "unknown"

    elif trial_id.startswith("hthora"):
        species = "H. thoracica"
        m = re.match(r"hthora_?(\d+)", trial_id)
        animal_no = int(m.group(1)) if m else 0
        sex = "unknown"

    else:
        species = "unknown"
        animal_no = 0
        sex = "unknown"

    return {"species": species, "sex": sex, "animal_no": animal_no}


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_all_trials(processed_dir: str, fps: float):
    """Load all processed trials and compute per-frame speeds.

    Returns a list of dicts, one per trial.
    """
    # Find all _center_mm.npy files
    files = sorted([f for f in os.listdir(processed_dir)
                    if f.endswith("_center_mm.npy")])

    trials = []
    for f in files:
        trial_id = f.replace("_center_mm.npy", "")
        center_path = os.path.join(processed_dir, f)
        temp_path = os.path.join(processed_dir, f"{trial_id}_animal_temperature.npy")

        center_mm = np.load(center_path)  # (N, 2)

        if not os.path.isfile(temp_path):
            print(f"  [SKIP] {trial_id}: no temperature data")
            continue

        animal_temp = np.load(temp_path)  # (N,)

        # Compute speed: mm/s
        dx = np.diff(center_mm[:, 0])
        dy = np.diff(center_mm[:, 1])
        speed_mms = np.sqrt(dx**2 + dy**2) * fps  # mm per second
        # Pad to same length as position (first frame has no speed)
        speed_mms = np.concatenate([[np.nan], speed_mms])

        meta = parse_trial_id(trial_id)

        # Align lengths (should be same, but be safe)
        n = min(len(center_mm), len(animal_temp), len(speed_mms))

        trials.append({
            "trial_id": trial_id,
            **meta,
            "center_mm": center_mm[:n],
            "temperature": animal_temp[:n],
            "speed_mms": speed_mms[:n],
            "n_frames": n,
        })

    print(f"[INFO] Loaded {len(trials)} trials with trajectory + temperature data.\n")
    return trials


# ──────────────────────────────────────────────────────────────────────────────
# 1. Step histograms
# ──────────────────────────────────────────────────────────────────────────────

def plot_histograms(trials: list, output_dir: str):
    """Step histograms of speed (>2 mm/s) and temperature, per species.
    Also writes a long-form CSV with all per-frame data."""

    # ── Collect per-frame data for CSV ──
    csv_rows = []
    species_speed = defaultdict(list)
    species_temp = defaultdict(list)

    for t in trials:
        sp = t["species"]
        sex = t["sex"]
        ano = t["animal_no"]

        valid = ~np.isnan(t["speed_mms"]) & ~np.isnan(t["temperature"])
        speeds = t["speed_mms"][valid]
        temps = t["temperature"][valid]

        # For CSV: all frames that have valid data, speed > threshold
        for s, tmp in zip(speeds, temps):
            if s > SPEED_THRESHOLD_MMS:
                csv_rows.append([sp, sex, ano, round(s, 4), round(tmp, 2)])

        # For speed histogram: only above threshold
        species_speed[sp].extend(speeds[speeds > SPEED_THRESHOLD_MMS].tolist())
        # For temperature histogram: everything
        species_temp[sp].extend(temps.tolist())

    # ── Write CSV ──
    csv_path = os.path.join(output_dir, "histogram_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "sex", "animal_no", "speed_mm/s", "temperature_degC"])
        w.writerows(csv_rows)
    print(f"  CSV → {csv_path}  ({len(csv_rows)} rows)")

    # ── Speed histogram ──
    fig, ax = plt.subplots(figsize=(8, 4))
    for sp in SPECIES_ORDER:
        if sp in species_speed and len(species_speed[sp]) > 0:
            data = np.array(species_speed[sp])
            bins = np.linspace(SPEED_THRESHOLD_MMS, np.percentile(data, 99.5), 60)
            ax.hist(data, bins=bins, histtype="step", linewidth=1.5,
                    density=True, label=sp, color=SPECIES_COLOURS[sp])
    ax.set_xlabel("Speed (mm/s)")
    ax.set_ylabel("Density")
    ax.set_title("Speed distribution (> 2 mm/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "histogram_speed.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "histogram_speed.svg"))
    plt.close(fig)
    print(f"  Plot → histogram_speed.png/.svg")

    # ── Temperature histogram ──
    fig, ax = plt.subplots(figsize=(8, 4))
    for sp in SPECIES_ORDER:
        if sp in species_temp and len(species_temp[sp]) > 0:
            data = np.array(species_temp[sp])
            bins = np.linspace(data.min(), data.max(), 60)
            ax.hist(data, bins=bins, histtype="step", linewidth=1.5,
                    density=True, label=sp, color=SPECIES_COLOURS[sp])
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Density")
    ax.set_title("Temperature distribution at animal position")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "histogram_temperature.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "histogram_temperature.svg"))
    plt.close(fig)
    print(f"  Plot → histogram_temperature.png/.svg")


# ──────────────────────────────────────────────────────────────────────────────
# 2. Boxplots
# ──────────────────────────────────────────────────────────────────────────────

def plot_boxplots(trials: list, output_dir: str):
    """Boxplots of median speed and median temperature per animal.
    Writes a summary CSV with one row per animal."""

    # ── Compute per-animal medians ──
    csv_rows = []
    species_medspeed = defaultdict(list)
    species_medtemp = defaultdict(list)

    for t in trials:
        sp = t["species"]
        sex = t["sex"]

        valid_s = t["speed_mms"][~np.isnan(t["speed_mms"])]
        valid_s = valid_s[valid_s > SPEED_THRESHOLD_MMS]
        med_speed = float(np.median(valid_s)) if len(valid_s) > 0 else np.nan

        valid_t = t["temperature"][~np.isnan(t["temperature"])]
        med_temp = float(np.median(valid_t)) if len(valid_t) > 0 else np.nan

        csv_rows.append([sp, sex, round(med_speed, 4), round(med_temp, 2)])
        species_medspeed[sp].append(med_speed)
        species_medtemp[sp].append(med_temp)

    # ── Write CSV ──
    csv_path = os.path.join(output_dir, "boxplot_data.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["species", "sex", "median_speed_mm/s", "median_temperature_degC"])
        w.writerows(csv_rows)
    print(f"  CSV → {csv_path}  ({len(csv_rows)} rows)")

    # ── Speed boxplot ──
    fig, ax = plt.subplots(figsize=(5, 4))
    bp_data, bp_labels, bp_colours = [], [], []
    for sp in SPECIES_ORDER:
        vals = [v for v in species_medspeed[sp] if not np.isnan(v)]
        if vals:
            bp_data.append(vals)
            # Use matplotlib mathtext for italic genus + species
            genus, epithet = sp.split(". ")
            bp_labels.append(f"$\\it{{{genus}.\\ {epithet}}}$")
            bp_colours.append(SPECIES_COLOURS[sp])

    bplot = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                       widths=0.5, showfliers=True)
    for patch, col in zip(bplot["boxes"], bp_colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    for patch, col in zip(bplot["medians"], bp_colours):
        patch.set_color("k")
    # Overlay individual points
    for i, (vals, col) in enumerate(zip(bp_data, bp_colours)):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, color=col, edgecolors="k", linewidths=0.5,
                   s=30, zorder=5)
    ax.set_ylabel("Median speed (mm/s)")
    ax.set_title("Median speed per animal (> 2 mm/s)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_speed.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "boxplot_speed.svg"))
    plt.close(fig)
    print(f"  Plot → boxplot_speed.png/.svg")

    # ── Temperature boxplot ──
    fig, ax = plt.subplots(figsize=(5, 4))
    bp_data, bp_labels, bp_colours = [], [], []
    for sp in SPECIES_ORDER:
        vals = [v for v in species_medtemp[sp] if not np.isnan(v)]
        if vals:
            bp_data.append(vals)
            genus, epithet = sp.split(". ")
            bp_labels.append(f"$\\it{{{genus}.\\ {epithet}}}$")
            bp_colours.append(SPECIES_COLOURS[sp])

    bplot = ax.boxplot(bp_data, labels=bp_labels, patch_artist=True,
                       widths=0.5, showfliers=True)
    for patch, col in zip(bplot["boxes"], bp_colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    for patch, col in zip(bplot["medians"], bp_colours):
        patch.set_color("k")
    for i, (vals, col) in enumerate(zip(bp_data, bp_colours)):
        x = np.random.normal(i + 1, 0.04, size=len(vals))
        ax.scatter(x, vals, color=col, edgecolors="k", linewidths=0.5,
                   s=30, zorder=5)
    ax.set_ylabel("Median temperature (°C)")
    ax.set_title("Median temperature at animal position")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "boxplot_temperature.png"), dpi=200)
    fig.savefig(os.path.join(output_dir, "boxplot_temperature.svg"))
    plt.close(fig)
    print(f"  Plot → boxplot_temperature.png/.svg")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Heatmaps
# ──────────────────────────────────────────────────────────────────────────────

def plot_heatmaps(trials: list, output_dir: str):
    """Positional (x, y) and thermal (T, y) filled contour plots per species.

    Positional: 0–600 mm × 0–80 mm, one per species, shared colour scale.
    Thermal:    global T range × 0–80 mm, one per species, shared colour scale.
    """
    from scipy.ndimage import gaussian_filter

    # ── Determine global temperature range across ALL species ──
    all_temps = []
    for t in trials:
        valid = t["temperature"][~np.isnan(t["temperature"])]
        if len(valid) > 0:
            all_temps.append(valid)
    all_temps = np.concatenate(all_temps)
    t_min = np.floor(all_temps.min())
    t_max = np.ceil(all_temps.max())
    print(f"  Global temperature range: {t_min:.0f}–{t_max:.0f} °C")

    # ── Group by species ──
    species_data = defaultdict(lambda: {"x": [], "y": [], "temp": []})
    for t in trials:
        sp = t["species"]
        valid = ~np.isnan(t["center_mm"][:, 0]) & ~np.isnan(t["temperature"])
        species_data[sp]["x"].extend(t["center_mm"][valid, 0].tolist())
        species_data[sp]["y"].extend(t["center_mm"][valid, 1].tolist())
        species_data[sp]["temp"].extend(t["temperature"][valid].tolist())

    # ── Bin settings (coarser to avoid sparsity) ──
    x_bins = np.linspace(0, 600, 31)       # 20 mm resolution
    y_bins = np.linspace(0, 80, 9)         # 10 mm resolution
    t_bins = np.linspace(t_min, t_max, 31) # ~1 °C resolution

    # Bin centres for contour grids
    x_centers = 0.5 * (x_bins[:-1] + x_bins[1:])
    y_centers = 0.5 * (y_bins[:-1] + y_bins[1:])
    t_centers = 0.5 * (t_bins[:-1] + t_bins[1:])

    sigma = 1.2  # Gaussian smoothing (in bin units)

    # ── First pass: compute all density grids to find global maxima ──
    pos_grids = {}
    temp_grids = {}

    for sp in SPECIES_ORDER:
        if sp not in species_data or len(species_data[sp]["x"]) == 0:
            continue

        x = np.array(species_data[sp]["x"])
        y = np.array(species_data[sp]["y"])
        temp = np.array(species_data[sp]["temp"])

        # Positional
        H_pos, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        H_pos = H_pos.T  # (y, x)
        H_pos_norm = H_pos / H_pos.sum() if H_pos.sum() > 0 else H_pos
        H_pos_smooth = gaussian_filter(H_pos_norm, sigma=sigma)
        pos_grids[sp] = H_pos_smooth

        # Thermal
        H_temp, _, _ = np.histogram2d(temp, y, bins=[t_bins, y_bins])
        H_temp = H_temp.T  # (y, t)
        H_temp_norm = H_temp / H_temp.sum() if H_temp.sum() > 0 else H_temp
        H_temp_smooth = gaussian_filter(H_temp_norm, sigma=sigma)
        temp_grids[sp] = H_temp_smooth

    # Global vmax for shared colour axes
    pos_vmax = max(g.max() for g in pos_grids.values()) if pos_grids else 1
    temp_vmax = max(g.max() for g in temp_grids.values()) if temp_grids else 1

    n_contour_levels = 20

    # ── Second pass: plot ──
    for sp in SPECIES_ORDER:
        if sp not in pos_grids:
            continue

        sp_label = sp.replace("H. ", "H_")

        # ── Positional contour ──
        fig, ax = plt.subplots(figsize=(10, 2.5))
        levels = np.linspace(0, pos_vmax, n_contour_levels + 1)
        cf = ax.contourf(x_centers, y_centers, pos_grids[sp],
                         levels=levels, cmap="hot", extend="max")
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Density")
        ax.set_xlim(0, 600)
        ax.set_ylim(0, 80)
        ax.set_xlabel("x position (mm)")
        ax.set_ylabel("y position (mm)")
        ax.set_title(f"Positional density — {sp}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"heatmap_position_{sp_label}.png"), dpi=200)
        fig.savefig(os.path.join(output_dir, f"heatmap_position_{sp_label}.svg"))
        plt.close(fig)

        # ── Temperature contour ──
        fig, ax = plt.subplots(figsize=(10, 2.5))
        levels = np.linspace(0, temp_vmax, n_contour_levels + 1)
        cf = ax.contourf(t_centers, y_centers, temp_grids[sp],
                         levels=levels, cmap="hot", extend="max")
        cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label("Density")
        ax.set_xlim(t_min, t_max)
        ax.set_ylim(0, 80)
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("y position (mm)")
        ax.set_title(f"Temperature density — {sp}")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"heatmap_temperature_{sp_label}.png"), dpi=200)
        fig.savefig(os.path.join(output_dir, f"heatmap_temperature_{sp_label}.svg"))
        plt.close(fig)

        print(f"  Contour plots → heatmap_position_{sp_label}.png/.svg, "
              f"heatmap_temperature_{sp_label}.png/.svg")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Weta trajectory + temperature analysis plots."
    )
    parser.add_argument(
        "--base_dir", type=str,
        default="/home/geuba03p/weta_project/weta_videos_cropped",
    )
    parser.add_argument(
        "--processed_dir", type=str, default=None,
        help="Directory with _center_mm.npy and _animal_temperature.npy "
             "(default: <base_dir>/processed_trajectories).",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for plots and CSVs "
             "(default: <base_dir>/analysis_output).",
    )
    parser.add_argument("--fps", type=float, default=25.0)

    args = parser.parse_args()

    processed_dir = args.processed_dir or os.path.join(args.base_dir, "processed_trajectories")
    output_dir = args.output_dir or os.path.join(args.base_dir, "analysis_output")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Input  : {processed_dir}")
    print(f"Output : {output_dir}\n")

    trials = load_all_trials(processed_dir, args.fps)

    if not trials:
        print("[ERROR] No trials loaded. Check processed_dir.")
        return

    print("── 1. Histograms ──")
    plot_histograms(trials, output_dir)

    print("\n── 2. Boxplots ──")
    plot_boxplots(trials, output_dir)

    print("\n── 3. Heatmaps ──")
    plot_heatmaps(trials, output_dir)

    print(f"\n[DONE] All outputs in {output_dir}/")


if __name__ == "__main__":
    main()