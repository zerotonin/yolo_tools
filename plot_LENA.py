#!/usr/bin/env python3

"""
analyze_detectors.py

Combines:
  1) Aggregation of final metrics (mAP, etc.) from:
     - TorchVision-based (retinanet, fasterrcnn, fcos) => <model>_eval.csv
     - Ultralytics-based (yolo8, yolo11, rtdetr) => find latest run, parse <model>_summary.csv or results.csv
  2) Merging 'inference.csv' data for each model, if present, to get:
     - avg_detection_time_ms (per image)
     - inference_fps
  3) Produces bar charts for the chosen metrics (map_50, map_75, avg_detection_time_ms, inference_fps).
  4) Combines training-loss curves into a single line plot for all models:
     - TorchVision => <model>_losses.csv
     - Ultralytics => sum of 'train/*_loss' columns in results.csv

Usage:
  python analyze_detectors.py

Assumptions:
- TorchVision model folders:
    <BASE_DIR>/<model_folder>/
       ├─ <model>_eval.csv
       ├─ <model>_losses.csv (if you have training loss)
       ├─ <model>_inference.csv (optional if you measure inference time)
- Ultralytics model folders:
    <BASE_DIR>/<model_runX>/ 
       ├─ results.csv
       ├─ <model>_summary.csv (maybe)
       ├─ inference.csv (if you measured inference time with a separate script)

Then it merges these so you can plot e.g. map_50, map_75, avg_detection_time_ms, inference_fps, etc.
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# #############################################################################
# CONFIG
# #############################################################################
BASE_DIR = "/media/aoraki_home/object_det_comparison"  # root folder with subdirectories
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# We'll produce bar charts for these metrics:
TARGET_METRICS = ["map_50", "map_75", "avg_detection_time_ms", "inference_fps"]

MODEL_CONFIG = {
    # TorchVision-based
    "fasterrcnn": {
        "type": "torchvision",
        "folder": "fasterrcnn",
        "eval_csv": "fasterrcnn_eval.csv",
        "loss_csv": "fasterrcnn_losses.csv",
        "inference_csv": "fasterrcnn_inference.csv",  # if it exists
        "display_name": "FasterRCNN",
    },
    "retinanet": {
        "type": "torchvision",
        "folder": "retinanet",
        "eval_csv": "retinanet_eval.csv",
        "loss_csv": "retinanet_losses.csv",
        "inference_csv": "retinanet_inference.csv",  # if it exists
        "display_name": "RetinaNet",
    },
    "fcos": {
        "type": "torchvision",
        "folder": "fcos",
        "eval_csv": "fcos_eval.csv",
        "loss_csv": "fcos_losses.csv",
        "inference_csv": "fcos_inference.csv",  # if it exists
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

# If some columns appear under different names:
RENAME_MAP = {
    "metrics/mAP50(B)": "map_50",
    "metrics/mAP50-95(B)": "map",
    # ...
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
    """Rename columns in row_dict according to RENAME_MAP."""
    new_dict = {}
    for k, v in row_dict.items():
        if k in RENAME_MAP:
            new_k = RENAME_MAP[k]
        else:
            new_k = k
        new_dict[new_k] = v
    return new_dict

# -------------------------------
# Parsing evaluation CSV
# -------------------------------
def parse_torchvision_eval(cfg):
    """
    For TorchVision: read <folder>/<eval_csv>.
    We'll take the first row, rename columns, return as a dict.
    """
    path = os.path.join(BASE_DIR, cfg["folder"], cfg["eval_csv"])
    if not os.path.isfile(path):
        print(f"[TorchVision eval] Missing: {path}")
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    return rename_columns(row)

def parse_ultra_eval(cfg):
    """
    For Ultralytics: find newest run folder, parse summary CSV or last row of results.csv, rename columns.
    """
    run_folder = find_latest_ultra_run(BASE_DIR, cfg["run_prefix"])
    if not run_folder:
        print(f"[Ultralytics eval] No run folder for {cfg['run_prefix']}")
        return {}
    # summary first
    sum_path = os.path.join(run_folder, cfg["summary_csv"])
    if os.path.isfile(sum_path):
        df = pd.read_csv(sum_path)
        if not df.empty:
            row = df.iloc[-1].to_dict()
            return rename_columns(row)
    # fallback to results.csv
    res_path = os.path.join(run_folder, "results.csv")
    if os.path.isfile(res_path):
        df = pd.read_csv(res_path)
        if not df.empty:
            row = df.iloc[-1].to_dict()
            return rename_columns(row)

    print(f"[Ultralytics eval] No summary or results.csv for {run_folder}")
    return {}

# -------------------------------
# Parsing training loss
# -------------------------------
def parse_torchvision_loss(cfg):
    """
    TorchVision: <folder>/<loss_csv> => columns [epoch, loss].
    """
    if "loss_csv" not in cfg:
        return pd.DataFrame()
    path = os.path.join(BASE_DIR, cfg["folder"], cfg["loss_csv"])
    if not os.path.isfile(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "epoch" not in df.columns or "loss" not in df.columns:
        return pd.DataFrame()
    df["model"] = cfg["display_name"]
    return df[["epoch", "loss", "model"]]

def parse_ultra_loss(cfg):
    """
    Ultralytics: sum up train/*_loss in results.csv => single 'loss' column
    """
    run_folder = find_latest_ultra_run(BASE_DIR, cfg["run_prefix"])
    if not run_folder:
        return pd.DataFrame()
    res_path = os.path.join(run_folder, "results.csv")
    if not os.path.isfile(res_path):
        return pd.DataFrame()
    df = pd.read_csv(res_path)
    if "epoch" not in df.columns:
        return pd.DataFrame()
    # find columns like train/cls_loss, train/box_loss...
    loss_cols = [c for c in df.columns if c.startswith("train/") and c.endswith("_loss")]
    if not loss_cols:
        return pd.DataFrame()
    df["train_loss"] = df[loss_cols].sum(axis=1)
    out = df[["epoch", "train_loss"]].copy()
    out.rename(columns={"train_loss": "loss"}, inplace=True)
    out["model"] = cfg["display_name"]
    return out[["epoch", "loss", "model"]]

# -------------------------------
# Parsing inference CSV
# -------------------------------
def parse_torchvision_inference(cfg):
    """
    For TorchVision: <folder>/<inference_csv> 
    We'll read the first row if it exists, expecting columns like:
      total_ms => we rename to avg_detection_time_ms
      fps => we rename to inference_fps
    """
    if "inference_csv" not in cfg:
        return {}
    path = os.path.join(BASE_DIR, cfg["folder"], cfg["inference_csv"])
    if not os.path.isfile(path):
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    # rename total_ms -> avg_detection_time_ms, fps -> inference_fps if they exist
    out = {}
    if "total_ms" in row:
        out["avg_detection_time_ms"] = row["total_ms"]
    if "fps" in row:
        out["inference_fps"] = row["fps"]
    return out

def parse_ultra_inference(cfg):
    """
    For Ultralytics: we expect an 'inference.csv' in the run folder, 
    containing columns like total_ms, fps, etc. 
    We'll read the first row if present and rename them to avg_detection_time_ms, inference_fps.
    """
    run_folder = find_latest_ultra_run(BASE_DIR, cfg["run_prefix"])
    if not run_folder:
        return {}
    inf_path = os.path.join(run_folder, "inference.csv")
    if not os.path.isfile(inf_path):
        return {}
    df = pd.read_csv(inf_path)
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    out = {}
    if "total_ms" in row:
        out["avg_detection_time_ms"] = row["total_ms"]
    if "fps" in row:
        out["inference_fps"] = row["fps"]
    return out

# #############################################################################
# MAIN
# #############################################################################
def main():
    # Step 1) Aggregate final metrics
    rows = []
    for model_key, cfg in MODEL_CONFIG.items():
        # parse final evaluation
        if cfg["type"] == "torchvision":
            eval_dict = parse_torchvision_eval(cfg)
        else:
            eval_dict = parse_ultra_eval(cfg)

        # parse inference time
        if cfg["type"] == "torchvision":
            inf_dict = parse_torchvision_inference(cfg)
        else:
            inf_dict = parse_ultra_inference(cfg)

        # combine them
        row = {}
        row.update(eval_dict)
        row.update(inf_dict)

        row["model"] = cfg.get("display_name", model_key)
        rows.append(row)

    df_all = pd.DataFrame(rows)
    # reorder columns
    cols = ["model"] + [c for c in df_all.columns if c != "model"]
    df_all = df_all[cols]

    # Save aggregated
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
        df_plot.sort_values(by=metric, ascending=False, inplace=True)

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
    all_losses = []
    for model_key, cfg in MODEL_CONFIG.items():
        if cfg["type"] == "torchvision":
            df_loss = parse_torchvision_loss(cfg)
        else:
            df_loss = parse_ultra_loss(cfg)

        if not df_loss.empty:
            all_losses.append(df_loss)

    if all_losses:
        df_loss_all = pd.concat(all_losses, ignore_index=True)
        # We'll plot each model as a separate line: epoch vs. loss
        plt.figure(figsize=(8,6))
        # group by model
        for model_name, dfg in df_loss_all.groupby("model"):
            dfg = dfg.sort_values("epoch")
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
