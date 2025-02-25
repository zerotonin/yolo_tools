#!/usr/bin/env python3

import argparse
import os
import pandas as pd
from ultralytics import YOLO

"""
eval_inference_speed.py

Measures inference speed on the validation set for an Ultralytics YOLO model 
(e.g. YOLOv8, YOLOv11, RT-DETR) by loading the best.pt (or other checkpoint) 
and running model.val() on the dataset. Saves timing + final metrics to a CSV.

Usage Example:
  python eval_inference_speed.py \
    --weights /home/user/object_det_comparison/charon_run/weights/best.pt \
    --data /projects/sciences/zoology/geurten_lab/AI_trainData/mydata.yaml \
    --batch 8 \
    --device 0 \
    --out_csv /path/to/inference_speed.csv \
    --name YOLOv8-l

Then you'll get a single-row CSV with columns like:
  model, preprocess_ms, inference_ms, postprocess_ms, total_ms, fps, (any mAP metrics)
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate inference speed on the YOLO validation set.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to best.pt or other checkpoint. E.g. /path/to/best.pt")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data.yaml referencing the same val set used for training.")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for validation.")
    parser.add_argument("--device", type=str, default="0",
                        help="Which device to use: e.g. '0' for GPU 0, 'cpu' for CPU, etc.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Validation image size. Usually matches training.")
    parser.add_argument("--half", action="store_true",
                        help="If set, use FP16 half precision for inference (GPU only).")
    parser.add_argument("--out_csv", type=str, default="inference_speed.csv",
                        help="Path to output CSV file for saving results.")
    parser.add_argument("--name", type=str, default="model",
                        help="Model name or label for the CSV row, e.g. YOLOv8-l.")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading model weights: {args.weights}")
    model = YOLO(args.weights)

    print("Running model.val() to measure inference speed on validation set...")
    results = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half
    )

    # Prepare a dictionary to store everything
    row_dict = {}
    row_dict["model"] = args.name  # your custom label

    # results.speed is typically a dict with keys:
    #   'preprocess' (ms per image),
    #   'inference' (ms per image),
    #   'postprocess' (ms per image)
    speed_info = getattr(results, "speed", None)
    if speed_info is not None:
        preprocess = speed_info.get("preprocess", 0.0)
        inference  = speed_info.get("inference", 0.0)
        postprocess= speed_info.get("postprocess", 0.0)
        total_ms   = preprocess + inference + postprocess
        fps        = 1000.0 / total_ms if total_ms > 0 else 0.0

        row_dict["preprocess_ms"]   = float(preprocess)
        row_dict["inference_ms"]    = float(inference)
        row_dict["postprocess_ms"]  = float(postprocess)
        row_dict["total_ms"]        = float(total_ms)
        row_dict["fps"]             = float(fps)
    else:
        print("No speed info found in results.speed. Possibly older or different version of Ultralytics?")

    # If you want to store final mAP metrics or others
    if hasattr(results, "metrics") and isinstance(results.metrics, dict):
        for k, v in results.metrics.items():
            # Convert to float if single scalar
            if isinstance(v, (float, int)):
                row_dict[k] = float(v)
            elif hasattr(v, "item"):
                row_dict[k] = float(v.item())
            else:
                # Could store as string, etc.
                row_dict[k] = v
    else:
        print("No .metrics found in results.")

    # Write out a single-row CSV
    df_out = pd.DataFrame([row_dict])
    out_path = args.out_csv
    df_out.to_csv(out_path, index=False)
    print(f"Saved inference speed + metrics to {out_path}")

    print("\n==== Summary ====")
    for k, val in row_dict.items():
        print(f"{k}: {val}")

if __name__ == "__main__":
    main()

# python eval_inference_speed.py \
#   --weights /media/aoraki_home/object_det_comparison/rtdetr_run/weights/best.pt\
#   --data /media/GLAB/AI_trainData/uLytics_Lena/data_4Bart.yml \
#   --batch 8 \
#   --device 0 \
#   --imgsz 640 \
#   --name RTDETR \
#   --out_csv /media/aoraki_home/object_det_comparison/rtdetr_run/inference.csv