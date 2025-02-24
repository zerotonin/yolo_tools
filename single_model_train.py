#!/usr/bin/env python3

import argparse
import os
import json
import random
import time

import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image

# pip install pycocotools torchmetrics matplotlib tqdm pandas
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# TorchVision detection models
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn,
    FCOS_ResNet50_FPN_Weights
)
from torchvision.models.detection.fcos import FCOSClassificationHead, FCOSRegressionHead

###############################################################################
# CONFIG & ARGUMENT PARSING
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate a single TorchVision detection model."
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=["retinanet", "fasterrcnn", "fcos"],
                        help="Which model to train.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--data_dir", type=str, default=".",
                        help="Directory containing 'original_images' and 'annotations.json'.")
    parser.add_argument("--output_parent", type=str, default="all_models_comparison",
                        help="Parent folder where model outputs are saved.")
    parser.add_argument("--num_classes", type=int, default=3,
                        help="Number of classes, including background (if used).")
    parser.add_argument("--split_ratio", type=float, default=0.8,
                        help="Train/val split ratio.")
    return parser.parse_args()

###############################################################################
# DATASET
###############################################################################
def load_json_data(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

class CustomCocoDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, images_dir, transforms=None):
        super().__init__()
        self.images_info = data_dict["images"]
        self.annotations = data_dict["annotations"]
        self.categories = data_dict["categories"]
        self.images_dir = images_dir
        self.transforms = transforms

        self.image_id_to_anns = {}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, idx):
        img_info = self.images_info[idx]
        img_id = img_info["id"]
        img_path = os.path.join(self.images_dir, img_info["file_name"])
        image = Image.open(img_path).convert("RGB")
        anns = self.image_id_to_anns.get(img_id, [])

        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        image_id_tensor = torch.tensor([idx])
        area = (boxes_tensor[:, 3] - boxes_tensor[:, 1]) * (boxes_tensor[:, 2] - boxes_tensor[:, 0])
        iscrowd = torch.zeros((len(boxes_tensor),), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": image_id_tensor,
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

def get_transform(train=True):
    tfs = [torchvision.transforms.ToTensor()]
    if train:
        tfs.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(tfs)

###############################################################################
# TRAIN+EVAL FUNCTION
###############################################################################
def train_and_evaluate(
    model_name, model, train_loader, val_loader,
    device, output_folder, num_epochs
):
    """
    Train the given TorchVision model and evaluate it.

    Returns: dict of final results (mAP, training time, etc.)
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # 1) Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

    # 2) Training loop
    losses_per_epoch = []
    print(f"=== Training {model_name} ===")
    train_start = time.time()
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_sum = 0.0
        pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{num_epochs}")
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values())
            total_loss.backward()
            optimizer.step()

            epoch_loss_sum += total_loss.item()
            pbar.set_postfix(loss=float(total_loss.item()))
        
        avg_loss = epoch_loss_sum / len(train_loader)
        losses_per_epoch.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], {model_name} - Avg Loss: {avg_loss:.4f}")

    total_train_time = time.time() - train_start
    print(f"{model_name} training complete. Elapsed time: {total_train_time:.2f} s")

    # 3) Save the model
    model_save_path = os.path.join(output_folder, f"{model_name}_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"{model_name} weights saved to {model_save_path}")

    # 4) Evaluate mAP (class_metrics=False by default; set to True if needed)
    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
            metric_map.update(outputs_cpu, targets_cpu)

    eval_res = metric_map.compute()
    # Convert to Python floats where possible
    eval_res_dict = {}
    for k, v in eval_res.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            eval_res_dict[k] = float(v.item())
        else:
            eval_res_dict[k] = v

    # 5) Detection time / inference speed
    inference_start = time.time()
    num_images = 0
    for images, _ in val_loader:
        images = [img.to(device) for img in images]
        _ = model(images)
        num_images += len(images)
    inference_end = time.time()

    total_detection_time = inference_end - inference_start
    avg_detection_time = total_detection_time / num_images
    inference_fps = 1.0 / avg_detection_time

    # 6) Save evaluation + training stats to CSV
    eval_csv_path = os.path.join(output_folder, f"{model_name}_eval.csv")
    df_eval = pd.DataFrame([{
        "model": model_name,
        "num_epochs": num_epochs,
        "total_train_time_s": total_train_time,
        "map": eval_res_dict.get("map", None),
        "map_50": eval_res_dict.get("map_50", None),
        "map_75": eval_res_dict.get("map_75", None),
        "avg_detection_time_ms": avg_detection_time*1000,
        "inference_fps": inference_fps
    }])
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"Saved evaluation metrics to {eval_csv_path}")

    # 7) Save training loss curve
    losses_csv_path = os.path.join(output_folder, f"{model_name}_losses.csv")
    pd.DataFrame({
        "epoch": list(range(1, num_epochs+1)),
        "loss": losses_per_epoch
    }).to_csv(losses_csv_path, index=False)

    plt.figure(figsize=(8,6))
    plt.plot(range(1, num_epochs+1), losses_per_epoch, marker='o', label='Train Loss')
    plt.title(f"Training Loss Curve ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_folder, f"{model_name}_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()  # close the figure

    # Return final results
    return {
        "model": model_name,
        "train_time": total_train_time,
        "map": eval_res_dict.get("map", 0.0),
        "map_50": eval_res_dict.get("map_50", 0.0),
        "map_75": eval_res_dict.get("map_75", 0.0),
        "avg_det_time_ms": avg_detection_time*1000,
        "inference_fps": inference_fps
    }

###############################################################################
# MAIN
###############################################################################
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directory
    os.makedirs(args.output_parent, exist_ok=True)

    # Build dataset
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "original_images")
    annotations_path = os.path.join(data_dir, "annotations", "annotations.json")

    data_all = load_json_data(annotations_path)
    images_all = data_all["images"]
    random.shuffle(images_all)
    split_idx = int(len(images_all) * args.split_ratio)

    train_images = images_all[:split_idx]
    val_images = images_all[split_idx:]

    data_train = {
        "images": train_images,
        "annotations": [
            ann for ann in data_all["annotations"]
            if ann["image_id"] in {img["id"] for img in train_images}
        ],
        "categories": data_all["categories"],
    }
    data_val = {
        "images": val_images,
        "annotations": [
            ann for ann in data_all["annotations"]
            if ann["image_id"] in {img["id"] for img in val_images}
        ],
        "categories": data_all["categories"],
    }

    train_dataset = CustomCocoDataset(data_train, images_dir, transforms=get_transform(train=True))
    val_dataset   = CustomCocoDataset(data_val, images_dir, transforms=get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: tuple(zip(*batch))
    )

    # Decide which model to train
    model_name = args.model.lower().strip()
    if model_name == "retinanet":
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        # Adjust classification/regression heads
        in_feats = model.head.classification_head.conv[0][0].in_channels
        model.head.classification_head = (
            torchvision.models.detection.retinanet.RetinaNetClassificationHead(
                in_feats, num_anchors=9, num_classes=args.num_classes
            )
        )
        model.head.regression_head = (
            torchvision.models.detection.retinanet.RetinaNetRegressionHead(
                in_feats, num_anchors=9
            )
        )
        output_folder = os.path.join(args.output_parent, "retinanet")

    elif model_name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        in_features_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features_f, args.num_classes
        )
        output_folder = os.path.join(args.output_parent, "fasterrcnn")

    elif model_name == "fcos":
        model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
        in_channels = model.head.classification_head.conv[0].in_channels
        model.head.classification_head = FCOSClassificationHead(
            in_channels=in_channels,
            num_anchors=1,
            num_classes=args.num_classes,
            num_convs=4,
            prior_probability=0.01
        )
        model.head.regression_head = FCOSRegressionHead(
            in_channels=in_channels,
            num_anchors=1,
            num_convs=4
        )
        output_folder = os.path.join(args.output_parent, "fcos")
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    # Move model to GPU (if available)
    model.to(device)

    # Train & evaluate
    results_dict = train_and_evaluate(
        model_name=model_name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_folder=output_folder,
        num_epochs=args.num_epochs
    )

    # Print final results
    print("==== Final Results ====")
    print(results_dict)

if __name__ == "__main__":
    main()
# # 1) RetinaNet
# python single_model_train.py \
#   --model retinanet \
#   --data_dir /home/geuba03p/label_test/COCO_annotations_RetinaNet_MaskRCNN/ \
#   --output_parent all_models_comparison \
#   --num_classes 3 \
#   --batch_size 3 \
#   --num_workers 2 \
#   --num_epochs 2 \
#   --split_ratio 0.8

# # 2) Faster R-CNN
# python single_model_train.py \
#   --model fasterrcnn \
#   --data_dir /home/geuba03p/label_test/COCO_annotations_RetinaNet_MaskRCNN/ \
#   --output_parent all_models_comparison \
#   --num_classes 3 \
#   --batch_size 3 \
#   --num_workers 2 \
#   --num_epochs 2 \
#   --split_ratio 0.8

# # 3) FCOS
# python single_model_train.py \
#   --model fcos \
#   --data_dir /home/geuba03p/label_test/COCO_annotations_RetinaNet_MaskRCNN/ \
#   --output_parent all_models_comparison \
#   --num_classes 3 \
#   --batch_size 3 \
#   --num_workers 2 \
#   --num_epochs 2 \
#   --split_ratio 0.8
