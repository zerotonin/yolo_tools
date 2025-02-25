#!/usr/bin/env python3

"""
analyze_detectors.py

1) Aggregates final metrics for multiple object detectors into a single CSV:
   - TorchVision-based (retinanet, fasterrcnn, fcos) => <model>_eval.csv
   - Ultralytics-based (yolo8, yolo11, rtdetr) => find the latest run folder 
     (e.g. yolo8_run3), parse <model>_summary.csv or the last row of results.csv

2) Produces bar charts for a set of chosen metrics (map_50, map_75, etc.) 
   and saves them as PNG and SVG to a "figures/" subfolder.

3) Also combines each model's training-loss curve into a single line plot:
   - TorchVision => <model>_losses.csv with columns [epoch, loss].
   - Ultralytics => sum up "train/box_loss", "train/cls_loss", "train/dfl_loss", etc. from results.csv
   to create a single "train_loss" per epoch.
   Then we overlay these lines in a single plot (epoch vs. loss) for all models.

Usage:
  python analyze_detectors.py
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# #############################################################################
# CONFIG
# #############################################################################
BASE_DIR = "/media/aoraki_home/object_det_comparison"  # Adjust if needed
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Final metrics we want for bar charts
TARGET_METRICS = ["map_50", "map_75", "avg_detection_time_ms", "inference_fps"]

MODEL_CONFIG = {
    # TorchVision-based
    "fasterrcnn": {
        "type": "torchvision",
        "folder": "fasterrcnn",
        "eval_csv": "fasterrcnn_eval.csv",
        "loss_csv": "fasterrcnn_losses.csv",
        "display_name": "FasterRCNN",
    },
    "retinanet": {
        "type": "torchvision",
        "folder": "retinanet",
        "eval_csv": "retinanet_eval.csv",
        "loss_csv": "retinanet_losses.csv",
        "display_name": "RetinaNet",
    },
    "fcos": {
        "type": "torchvision",
        "folder": "fcos",
        "eval_csv": "fcos_eval.csv",
        "loss_csv": "fcos_losses.csv",
        "display_name": "FCOS",
    },
    # Ultralytics-based
    "yolo8": {
        "type": "ultralytics",
        "run_prefix": "yolo8_run",
        "summary_csv": "yolo8_summary.csv",
        "display_name": "YOLOv8",
    },
    "yolo11": {
        "type": "ultralytics",
        "run_prefix": "yolo11_run",
        "summary_csv": "yolo11_summary.csv",
        "display_name": "YOLOv11",
    },
    "rtdetr": {
        "type": "ultralytics",
        "run_prefix": "rtdetr_run",
        "summary_csv": "rtdetr_summary.csv",
        "display_name": "RT-DETR",
    },
}

# For renaming columns from TorchVision or Ultralytics so they unify
RENAME_MAP = {
    "metrics/mAP50(B)": "map_50",
    "metrics/mAP50-95(B)": "map",
    # etc...
}


# #############################################################################
# HELPER FUNCTIONS
# #############################################################################
def find_latest_ultra_run(base_dir, run_prefix):
    """
    Finds subfolders named run_prefix + digits, e.g. yolo8_run, yolo8_run2, ...
    Returns the absolute path of highest numeric suffix or None if none found.
    """
    pattern = re.compile(rf"^{re.escape(run_prefix)}(\d+)?$")
    candidates = []
    for entry in os.listdir(base_dir):
        fullpath = os.path.join(base_dir, entry)
        if os.path.isdir(fullpath):
            match = pattern.match(entry)
            if match:
                num_str = match.group(1)
                num = int(num_str) if num_str else 0
                candidates.append((num, entry))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return os.path.join(base_dir, candidates[-1][1])

def rename_columns(row_dict):
    """
    Rename columns in row_dict according to RENAME_MAP.
    """
    new_dict = {}
    for k, v in row_dict.items():
        if k in RENAME_MAP:
            new_k = RENAME_MAP[k]
        else:
            new_k = k
        new_dict[new_k] = v
    return new_dict

def parse_torchvision_eval(model_cfg):
    """
    Reads <folder>/<eval_csv>, picks first row, rename columns, return as dict.
    """
    path = os.path.join(BASE_DIR, model_cfg["folder"], model_cfg["eval_csv"])
    if not os.path.isfile(path):
        print(f"WARNING: TorchVision eval CSV missing: {path}")
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    row = rename_columns(row)
    return row

def parse_ultra_eval(model_cfg):
    """
    For ultralytics, find latest run folder, parse <model>_summary.csv or last row of results.csv.
    rename columns, return dict.
    """
    run_folder = find_latest_ultra_run(BASE_DIR, model_cfg["run_prefix"])
    if not run_folder:
        print(f"No run folder found for {model_cfg['run_prefix']}")
        return {}
    # summary first
    sum_path = os.path.join(run_folder, model_cfg["summary_csv"])
    if os.path.isfile(sum_path):
        df = pd.read_csv(sum_path)
        if not df.empty:
            row = df.iloc[-1].to_dict()
            row = rename_columns(row)
            return row
    # fallback results.csv
    res_path = os.path.join(run_folder, "results.csv")
    if os.path.isfile(res_path):
        df = pd.read_csv(res_path)
        if not df.empty:
            row = df.iloc[-1].to_dict()
            row = rename_columns(row)
            return row

    print(f"WARNING: No summary or results.csv for {run_folder}")
    return {}

def parse_torchvision_loss(model_cfg):
    """
    Reads <folder>/<loss_csv> => columns [epoch, loss].
    Returns a DataFrame with columns: epoch, loss, model_name for plotting.
    """
    path = os.path.join(BASE_DIR, model_cfg["folder"], model_cfg["loss_csv"])
    if not os.path.isfile(path):
        print(f"WARNING: TorchVision loss CSV missing: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "epoch" not in df.columns or "loss" not in df.columns:
        print(f"WARNING: {path} missing 'epoch' or 'loss' columns.")
        return pd.DataFrame()
    # Add a column for the model name
    df["model"] = model_cfg["display_name"]
    return df[["epoch", "loss", "model"]]

def parse_ultra_loss(model_cfg):
    """
    For ultralytics, parse the 'results.csv' from the latest run.
    We'll sum up the 'train/box_loss', 'train/cls_loss', 'train/dfl_loss' 
    or 'train/giou_loss', 'train/l1_loss' if they exist, to create 'train_loss'.
    Then we keep [epoch, train_loss].
    """
    run_folder = find_latest_ultra_run(BASE_DIR, model_cfg["run_prefix"])
    if not run_folder:
        return pd.DataFrame()
    results_path = os.path.join(run_folder, "results.csv")
    if not os.path.isfile(results_path):
        print(f"WARNING: no results.csv for {model_cfg['run_prefix']} run.")
        return pd.DataFrame()
    df = pd.read_csv(results_path)

    # Which columns might exist?
    # YOLOv8 typically: 'epoch', 'train/box_loss', 'train/cls_loss', 'train/dfl_loss'
    # RT-DETR sometimes: 'train/giou_loss', 'train/l1_loss', ...
    # We'll gather any that start with 'train/' and end with '_loss'
    loss_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    if "epoch" not in df.columns or not loss_cols:
        print(f"WARNING: {results_path} missing 'epoch' or train/*_loss columns.")
        return pd.DataFrame()
    # sum them
    df["train_loss"] = df[loss_cols].sum(axis=1)

    out = df[["epoch", "train_loss"]].copy()
    out.rename(columns={"train_loss": "loss"}, inplace=True)
    out["model"] = model_cfg["display_name"]
    return out[["epoch", "loss", "model"]]

# #############################################################################
# MAIN
# #############################################################################
def main():
    # Step 1) Aggregate final metrics
    rows = []
    for model_key, cfg in MODEL_CONFIG.items():
        if cfg["type"] == "torchvision":
            row = parse_torchvision_eval(cfg)
        else:
            row = parse_ultra_eval(cfg)
        row["model"] = cfg.get("display_name", model_key)
        rows.append(row)
    df_all = pd.DataFrame(rows)
    # reorder columns
    cols = ["model"] + [c for c in df_all.columns if c != "model"]
    df_all = df_all[cols]
    # Save
    out_csv = os.path.join(BASE_DIR, "all_metrics.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"Aggregated metrics saved to {out_csv}")

    # Step 2) Bar charts for each metric in TARGET_METRICS
    for metric in TARGET_METRICS:
        if metric not in df_all.columns:
            print(f"Skipping metric '{metric}' (not found).")
            continue
        df_plot = df_all.dropna(subset=[metric]).copy()
        if df_plot.empty:
            print(f"No data for metric '{metric}'. Skipping plot.")
            continue
        df_plot.sort_values(metric, ascending=False, inplace=True)

        plt.figure(figsize=(7,5))
        bars = plt.bar(df_plot["model"], df_plot[metric], edgecolor="black")
        for bar in bars:
            val = bar.get_height()
            plt.text(bar.get_x()+bar.get_width()/2, val+0.01*abs(val),
                     f"{val:.3f}", ha='center', va='bottom', fontsize=9)

        plt.title(f"{metric} Comparison")
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        png_path = os.path.join(FIGURES_DIR, f"{metric}_comparison.png")
        svg_path = os.path.join(FIGURES_DIR, f"{metric}_comparison.svg")
        plt.savefig(png_path, dpi=150)
        plt.savefig(svg_path)
        plt.close()
        print(f"Saved {metric} bar chart to {png_path} and {svg_path}")

    # Step 3) Combine training-loss curves into one figure
    # We parse each model's "loss_csv" if TorchVision, or sum up 'train/*_loss' if Ultralytics
    all_losses = []
    for model_key, cfg in MODEL_CONFIG.items():
        if cfg["type"] == "torchvision":
            if "loss_csv" in cfg and cfg["loss_csv"]:
                df_loss = parse_torchvision_loss(cfg)
                if not df_loss.empty:
                    all_losses.append(df_loss)
        else:
            # ultralytics
            df_loss = parse_ultra_loss(cfg)
            if not df_loss.empty:
                all_losses.append(df_loss)

    if all_losses:
        df_loss_all = pd.concat(all_losses, ignore_index=True)
        # We'll plot each model as a separate line: epoch vs. loss

        plt.figure(figsize=(8,6))
        # group by model
        for model_name, dfg in df_loss_all.groupby("model"):
            dfg = dfg.sort_values("epoch")  # ensure epoch ascending
            plt.plot(dfg["epoch"], dfg["loss"], label=model_name)

        plt.title("Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        out_png = os.path.join(FIGURES_DIR, "training_loss_comparison.png")
        out_svg = os.path.join(FIGURES_DIR, "training_loss_comparison.svg")
        plt.savefig(out_png, dpi=150)
        plt.savefig(out_svg)
        plt.close()
        print(f"Saved training loss comparison to {out_png} and {out_svg}")
    else:
        print("No training-loss CSV data found for any model. Skipping loss plot.")

    print("Done.")

if __name__ == "__main__":
    main()
