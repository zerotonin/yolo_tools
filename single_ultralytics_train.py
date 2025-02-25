#!/usr/bin/env python3

import argparse
import os
import time
import json
import pandas as pd
from ultralytics import YOLO, RTDETR

"""
Usage:
  python single_ultralytics_train.py --model rtdetr \
     --pretrained rtdetr-l.pt \
     --data mydata.yaml \
     --epochs 2 \
     --batch_size 4 \
     --imgsz 640 \
     --project results_ultralytics --name rtdetr_run
     
Similarly for:
  --model yolo8   --pretrained yolov8n.pt
  --model yolo11  --pretrained yolo11n.pt

Now supports `--resume` to continue training from a checkpoint.
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train ultralytics RTDETR, YOLO8, or YOLO11 on a custom dataset.")
    parser.add_argument("--model", type=str, required=True,
                        choices=["rtdetr", "yolo8", "yolo11"],
                        help="Which model to train.")
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path or name of the pretrained checkpoint (e.g. 'rtdetr-l.pt', 'yolov8n.pt', or 'my_folder/weights/last.pt')")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to data YAML file referencing train/val images, e.g. 'mydata.yaml'")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training.")
    parser.add_argument("--project", type=str, default="ultra_runs",
                        help="Folder where results (weights, logs) are saved.")
    parser.add_argument("--name", type=str, default="run",
                        help="Name of the particular run inside the project folder.")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for measuring inference time.")
    parser.add_argument("--inference_images", type=str, default="",
                        help="Optional path to images or directory for measuring inference speed.")
    parser.add_argument("--resume", action="store_true",
                        help="If set, resume training from the specified --pretrained checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    print(f"==> Training {args.model} with pretrained={args.pretrained}, resume={args.resume}")

    # 1. Load the model
    if args.model.lower() == "rtdetr":
        model = RTDETR(args.pretrained)
    else:
        model = YOLO(args.pretrained)

    # 2. Model info (optional)
    try:
        model.info()
    except:
        pass

    # 3. Train (with resume)
    train_start = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        project=args.project,
        name=args.name,
        resume=args.resume,     # <--- key line to allow training resume
        # Additional arguments if needed (e.g., device=0, etc.)
    )
    train_time = time.time() - train_start
    print(f"Training complete. Elapsed time: {train_time:.2f}s")

    # 4. Basic info from training results
    metrics_dict = {}
    if hasattr(results, "metrics") and isinstance(results.metrics, dict):
        for k, v in results.metrics.items():
            if isinstance(v, (float, int)):
                metrics_dict[k] = float(v)
            elif hasattr(v, "item"):
                metrics_dict[k] = float(v.item())
            else:
                metrics_dict[k] = v

    # 5. Inference Speed Check
    if args.inference_images:
        inf_path = args.inference_images
    else:
        inf_path = os.path.join(args.project, args.name, "val_batch0_pred.jpg")

    t0 = time.time()
    predictions = model.predict(
        source=inf_path,
        conf=args.conf,
        save=False
    )
    t1 = time.time()
    inf_time_s = t1 - t0

    metrics_dict["inference_time_s"] = inf_time_s
    metrics_dict["training_time_s"] = train_time
    metrics_dict["model"] = args.model
    metrics_dict["pretrained"] = args.pretrained
    metrics_dict["epochs"] = args.epochs
    metrics_dict["batch_size"] = args.batch_size
    metrics_dict["imgsz"] = args.imgsz

    # 6. Save metrics to CSV
    df = pd.DataFrame([metrics_dict])
    out_csv = os.path.join(args.project, args.name, f"{args.model}_summary.csv")
    df.to_csv(out_csv, index=False)
    print(f"Saved final metrics to {out_csv}")

    print("==== Done ====")
    print(metrics_dict)

if __name__ == "__main__":
    main()
