import os
import json
import random
import time

import torch
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image

# Extra packages you need to install:
#   pip install pycocotools torchmetrics matplotlib tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# For the RetinaNet model
from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights
)

################################################################################
# 1. CONFIGURATION
################################################################################
root_dir = "/home/geuba03p/label_test/COCO_annotations_RetinaNet_MaskRCNN/"
images_dir = os.path.join(root_dir, "original_images")       # adjust as needed
annotations_path = os.path.join(os.path.join(root_dir,'annotations'), "annotations.json")  # single COCO annotations file

num_classes = 3  # 1 background + 2 object classes
batch_size = 4
num_workers = 4
num_epochs = 10   # for demo; increase if you want a more thorough training
split_ratio = 0.8  # 80% train, 20% val
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
# 2. DATASET DEFINITION
################################################################################
def load_json_data(json_path):
    """Load COCO-style JSON annotations."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return None

class CustomCocoDataset(torch.utils.data.Dataset):
    """
    A custom dataset for COCO-format data.
    Expects a dictionary with keys: ["images", "annotations", "categories"].
    The images live in `images_dir`.
    """
    def __init__(self, data_dict, images_dir, transforms=None):
        super().__init__()
        self.images_info = data_dict["images"]
        self.annotations = data_dict["annotations"]
        self.categories = data_dict["categories"]
        self.images_dir = images_dir
        self.transforms = transforms

        # Build a mapping from image_id -> list of annotations
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

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Get all annotations for this image
        anns = self.image_id_to_anns.get(img_id, [])

        # Convert [x, y, w, h] -> [x_min, y_min, x_max, y_max]
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
    """ Return a basic set of transforms. """
    transform_list = [torchvision.transforms.ToTensor()]
    if train:
        transform_list.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transform_list)

################################################################################
# 3. TRAIN/VAL SPLIT
################################################################################
data_all = load_json_data(annotations_path)
if data_all is None:
    raise FileNotFoundError(f"Could not load {annotations_path}")

images_all = data_all["images"]
random.shuffle(images_all)  # Shuffle in-place
split_idx = int(len(images_all) * split_ratio)
train_images = images_all[:split_idx]
val_images = images_all[split_idx:]

# Build two data dictionaries
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
    batch_size=1,  # Typically 1 for evaluation/inference
    shuffle=False,
    num_workers=num_workers,
    collate_fn=lambda batch: tuple(zip(*batch))
)

################################################################################
# 4. MODEL SETUP
################################################################################
model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
# Adjust classification/regression heads
in_features = model.head.classification_head.conv[0][0].in_channels
model.head.classification_head = (
    torchvision.models.detection.retinanet.RetinaNetClassificationHead(
        in_features,
        num_anchors=9,
        num_classes=num_classes
    )
)
model.head.regression_head = (
    torchvision.models.detection.retinanet.RetinaNetRegressionHead(
        in_features,
        num_anchors=9
    )
)
model.to(device)

################################################################################
# 5. TRAINING LOOP
################################################################################
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
num_epochs = num_epochs

losses_per_epoch = []  # store average loss each epoch for plotting

print("Starting training...")
train_start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    epoch_loss_sum = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for images, targets in progress_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()

        epoch_loss_sum += losses.item()
        progress_bar.set_postfix(loss=losses.item())
    
    avg_loss = epoch_loss_sum / len(train_loader)
    losses_per_epoch.append(avg_loss)
    print(f"Epoch [{epoch+1}/{num_epochs}] -- Average Loss: {avg_loss:.4f}")

total_train_time = time.time() - train_start_time
print(f"Training finished in {total_train_time:.2f} seconds.")

################################################################################
# 6. EVALUATION: MEAN AVERAGE PRECISION (mAP)
################################################################################
# We'll use torchmetrics' MeanAveragePrecision for simplicity.
metric_map = MeanAveragePrecision(
    box_format="xyxy", iou_type="bbox", class_metrics=False
)
# ^ class_metrics=False means we'll get an overall AP, not per-class.

model.eval()
with torch.no_grad():
    for images, targets in tqdm(val_loader, desc="Evaluating on val set"):
        images = [img.to(device) for img in images]
        outputs = model(images)
        # Move outputs/targets back to CPU for torchmetrics
        outputs_cpu = [{k: v.cpu() for k, v in out.items()} for out in outputs]
        targets_cpu = [{k: v.cpu() for k, v in tgt.items()} for tgt in targets]
        metric_map.update(outputs_cpu, targets_cpu)

eval_res = metric_map.compute()
# eval_res is a dict with keys like 'map', 'map_50', 'map_75', etc.
print("Validation metrics:")
for k, v in eval_res.items():
    print(f"{k}: {v:.4f}")

################################################################################
# 7. DETECTION TIME / INFERENCE SPEED
################################################################################
# We'll do a simple loop through the val_loader measuring average time per image
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

print(f"Avg detection time per image: {avg_detection_time*1000:.2f} ms")
print(f"Inference speed: {inference_fps:.2f} FPS")

################################################################################
# 8. SAVE THE MODEL
################################################################################
torch.save(model.state_dict(), "retinanet_trained_model.pth")
print("Model saved to retinanet_trained_model.pth")

################################################################################
# 9. PLOT THE TRAINING LOSS CURVE
################################################################################
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), losses_per_epoch, marker='o', label='Train Loss')
plt.title("Training Loss Curve (RetinaNet)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("train_loss_curve.png")
plt.show()

################################################################################
# SUMMARY PRINT
################################################################################
print("======= SUMMARY =======")
print(f"Total Training Time: {total_train_time:.2f} s")
print(f"Mean Average Precision (mAP): {eval_res['map']:.4f}")
print(f"Detection Time per image: {avg_detection_time*1000:.2f} ms")
print(f"Inference Speed: {inference_fps:.2f} FPS")
