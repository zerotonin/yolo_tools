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
from torchvision.models.detection import fcos_resnet50_fpn, FCOS_ResNet50_FPN_Weights

########################################################
# CONFIG
########################################################
root_dir = "/home/geuba03p/label_test/COCO_annotations_RetinaNet_MaskRCNN/"
images_dir = os.path.join(root_dir, "original_images")
annotations_path = os.path.join(root_dir, "annotations", "annotations.json")

num_classes = 3   # e.g. background + 2 objects
batch_size = 3
num_workers = 2
num_epochs = 50  # short for demo
split_ratio = 0.8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# We will store each model's results in a subfolder of this parent
output_parent = "all_models_comparison"
os.makedirs(output_parent, exist_ok=True)

########################################################
# DATASET
########################################################
def load_json_data(json_path):
    with open(json_path, 'r') as f:
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

########################################################
# TRAIN/VAL SPLIT
########################################################
data_all = load_json_data(annotations_path)
images_all = data_all["images"]
random.shuffle(images_all)
split_idx = int(len(images_all) * split_ratio)
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
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    collate_fn=lambda batch: tuple(zip(*batch))
)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=num_workers,
    collate_fn=lambda batch: tuple(zip(*batch))
)

########################################################
# TRAIN+EVAL FUNCTION
########################################################
def train_and_evaluate(model_name, model, output_folder):
    """Train the given TorchVision model and evaluate it.

    Args:
        model_name (str): A name identifier, e.g. 'retinanet', 'fasterrcnn', 'fcos', etc.
        model (nn.Module): A TorchVision detection model
        output_folder (str): Where to save logs, plots, and model.

    Returns:
        dict: A dictionary of final evaluation results (mAP, time, etc.)
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

    # 3) Save the model weights
    model_save_path = os.path.join(output_folder, f"{model_name}_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"{model_name} weights saved to {model_save_path}")

    # 4) Evaluate with class_metrics=True to get per-class stats
    metric_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
            metric_map.update(outputs_cpu, targets_cpu)

    eval_res = metric_map.compute()

    # Convert to Python floats, or JSON-serializable objects
    # (map_per_class, mar_100_per_class, etc. may be arrays)
    eval_res_dict = {}
    for k, v in eval_res.items():
        if isinstance(v, torch.Tensor) and v.numel() == 1:
            eval_res_dict[k] = float(v.item())
        elif isinstance(v, torch.Tensor):
            # e.g. map_per_class is a 1D Tensor -> convert to list
            eval_res_dict[k] = v.cpu().tolist()
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
    # We'll insert known top-level fields, plus detection time and so forth.
    eval_csv_path = os.path.join(output_folder, f"{model_name}_eval.csv")

    # If you know some typical fields from MeanAveragePrecision:
    #  map, map_50, map_75, map_small, map_medium, map_large,
    #  mar_1, mar_10, mar_100, mar_small, mar_medium, mar_large,
    #  map_per_class, mar_100_per_class
    # We'll just fetch them if they exist:
    def fetch_metric(key):
        return eval_res_dict.get(key, None)

    # Turn array fields (like map_per_class) into JSON so we can store them in one row
    def maybe_jsonify(val):
        if isinstance(val, list):
            return json.dumps(val)  # store arrays as JSON in the CSV cell
        return val

    row_dict = {
        "model": model_name,
        "num_epochs": num_epochs,
        "total_train_time_s": total_train_time,
        "map": fetch_metric("map"),
        "map_50": fetch_metric("map_50"),
        "map_75": fetch_metric("map_75"),
        "map_small": fetch_metric("map_small"),
        "map_medium": fetch_metric("map_medium"),
        "map_large": fetch_metric("map_large"),
        "mar_1": fetch_metric("mar_1"),
        "mar_10": fetch_metric("mar_10"),
        "mar_100": fetch_metric("mar_100"),
        "mar_small": fetch_metric("mar_small"),
        "mar_medium": fetch_metric("mar_medium"),
        "mar_large": fetch_metric("mar_large"),
        "map_per_class": maybe_jsonify(fetch_metric("map_per_class")),
        "mar_100_per_class": maybe_jsonify(fetch_metric("mar_100_per_class")),
        "avg_detection_time_ms": avg_detection_time * 1000,
        "inference_fps": inference_fps
    }

    df_eval = pd.DataFrame([row_dict])
    df_eval.to_csv(eval_csv_path, index=False)
    print(f"Saved evaluation metrics to {eval_csv_path}")

    # 7) Save training loss curve
    losses_csv_path = os.path.join(output_folder, f"{model_name}_losses.csv")
    df_losses = pd.DataFrame({
        "epoch": list(range(1, num_epochs + 1)),
        "loss": losses_per_epoch
    })
    df_losses.to_csv(losses_csv_path, index=False)

    plt.figure(figsize=(8,6))
    plt.plot(df_losses["epoch"], df_losses["loss"], marker='o', label='Train Loss')
    plt.title(f"Training Loss Curve ({model_name})")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_plot_path = os.path.join(output_folder, f"{model_name}_loss_curve.png")
    plt.savefig(loss_plot_path)
    plt.close()  # close the figure to avoid overlapping plots

    # Return a dict of final results (for the summary table)
    return row_dict

########################################################
# BUILD THREE MODELS & TRAIN EACH
########################################################
all_results = []  # we will accumulate a row of stats per model

# A) RetinaNet
retinanet = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
in_feats = retinanet.head.classification_head.conv[0][0].in_channels
retinanet.head.classification_head = (
    torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_feats, num_anchors=9, num_classes=num_classes
    )
)
retinanet.head.regression_head = (
    torchvision.models.detection.retinanet.RetinaNetRegressionHead(
        in_feats, num_anchors=9
    )
)
retinanet.to(device)

res_retina = train_and_evaluate(
    model_name="retinanet",
    model=retinanet,
    output_folder=os.path.join(output_parent, "retinanet")
)
all_results.append(res_retina)
time.sleep(10)

# B) Faster R-CNN (ResNet50-FPN)
fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
# Adjust num_classes in the box_predictor
in_features_f = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
fasterrcnn.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features_f, num_classes
)
fasterrcnn.to(device)

res_faster = train_and_evaluate(
    model_name="fasterrcnn",
    model=fasterrcnn,
    output_folder=os.path.join(output_parent, "fasterrcnn")
)
all_results.append(res_faster)
time.sleep(10)

# C) FCOS
# 1) Load the entire 91-class model
fcos = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.COCO_V1)
# 2) Overwrite the classification & regression heads
in_channels = fcos.head.classification_head.conv[0].in_channels
fcos.head.classification_head = FCOSClassificationHead(
    in_channels=in_channels,
    num_anchors=1,         # must specify =1 for FCOS
    num_classes=num_classes,
    num_convs=4,
    prior_probability=0.01
)
fcos.head.regression_head = FCOSRegressionHead(
    in_channels=in_channels,
    num_anchors=1,         # must specify
    num_convs=4
)
fcos.to(device)

res_fcos = train_and_evaluate(
    model_name="fcos",
    model=fcos,
    output_folder=os.path.join(output_parent, "fcos")
)
all_results.append(res_fcos)

########################################################
# SAVE COMPARATIVE RESULTS
########################################################
df_all = pd.DataFrame(all_results)
summary_csv = os.path.join(output_parent, "all_models_summary.csv")
df_all.to_csv(summary_csv, index=False)
print(f"==== Final Summary ====\n{df_all}\nSaved to {summary_csv}")
