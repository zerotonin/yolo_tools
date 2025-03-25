import torch, os
from ultralytics import YOLO
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, fcos_resnet50_fpn, retinanet_resnet50_fpn
)

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import cv2, time
import pandas as pd
import matplotlib.pyplot as plt

# Choose device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Number of object classes in dataset (excluding background) for TorchVision
NUM_CLASSES = 2  # e.g., if you have 2 foreground classes, +1 for background => 3 total
# If Charon_3 has 3 classes total, YOLO handles that internally—no extra background is needed.

###############################################################################
# Load TorchVision models with the correct num_classes
# Using weights=None ensures we do not inadvertently load a different architecture.
# strict=False lets us ignore minor mismatches in layer naming or shape.

# Faster R-CNN
fr_model = fasterrcnn_resnet50_fpn(weights=None, num_classes=NUM_CLASSES + 1)
fr_model.load_state_dict(
    torch.load("/home/geuba03p/all_models_comparison/fasterrcnn/fasterrcnn_weights.pth"),
    strict=False
)
fr_model.to(device).eval()

# FCOS
fcos_model = fcos_resnet50_fpn(weights=None, num_classes=NUM_CLASSES + 1)
fcos_model.load_state_dict(
    torch.load("/home/geuba03p/all_models_comparison/fcos/fcos_weights.pth"),
    strict=False
)
fcos_model.to(device).eval()

# RetinaNet
retina_model = retinanet_resnet50_fpn(weights=None, num_classes=NUM_CLASSES + 1)
retina_model.load_state_dict(
    torch.load("/home/geuba03p/all_models_comparison/retinanet/retinanet_weights.pth"),
    strict=False
)
retina_model.to(device).eval()

###############################################################################
# Load Ultralytics models (.pt format)
# These do not need the +1 background approach; YOLO always uses just #classes.

rtdetr_model  = YOLO("/home/geuba03p/all_models_comparison/rtdetr_run/weights/best.pt")
yolo8_model   = YOLO("/home/geuba03p/all_models_comparison/yolo8_run2/weights/best.pt")
yolo11_model  = YOLO("/home/geuba03p/all_models_comparison/yolo11_run4/weights/best.pt")
charon_model  = YOLO("/home/geuba03p/all_models_comparison/charon_run3/weights/best.pt")

###############################################################################
# Load COCO-style ground truth annotations
cocoGT = COCO("/path/to/instances_val.json")  # update with actual annotation path
img_ids = cocoGT.getImgIds()

###############################################################################
# Helper: run a TorchVision model on an image and format detections for COCOeval
def get_detections(model, image_path, image_id):
    import torchvision.transforms.functional as F
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Convert to tensor and normalize
    img_tensor = F.to_tensor(img_rgb)  # [C,H,W], float, 0..1
    img_tensor = F.normalize(img_tensor, mean=[0.485,0.456,0.406],
                                          std=[0.229,0.224,0.225])
    img_tensor = img_tensor.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    detections = []
    for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
        box = box.cpu().numpy()
        score = float(score.cpu().numpy())
        label = int(label.cpu().numpy())
        # Convert [x1, y1, x2, y2] -> [x, y, w, h]
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        detections.append({
            "image_id": image_id,
            "category_id": label,  # assuming label=category_id
            "bbox": [x1, y1, w, h],
            "score": score
        })
    return detections

###############################################################################
# Evaluate TorchVision models using COCOeval
model_detections = {"Faster R-CNN": [], "FCOS": [], "RetinaNet": []}
for img_id in img_ids:
    info = cocoGT.loadImgs(img_id)[0]
    img_path = os.path.join("/path/to/val_images/", info['file_name'])
    model_detections["Faster R-CNN"] += get_detections(fr_model,    img_path, img_id)
    model_detections["FCOS"]         += get_detections(fcos_model, img_path, img_id)
    model_detections["RetinaNet"]    += get_detections(retina_model, img_path, img_id)

metrics = {}
for model_name, dets in model_detections.items():
    cocoDT = cocoGT.loadRes(dets)
    cocoEval = COCOeval(cocoGT, cocoDT, iouType='bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    mAP50_95, AP50, AP75 = cocoEval.stats[0], cocoEval.stats[1], cocoEval.stats[2]
    metrics[model_name] = (mAP50_95, AP50, AP75)

###############################################################################
# Evaluate Ultralytics models using the built-in .val() method
# (Assumes each model contains or references the correct validation dataset.)
ultra_metrics = {}
for name, model in [
    ("RT-DETR", rtdetr_model),
    ("YOLOv8",  yolo8_model),
    ("YOLO11",  yolo11_model),
    ("Charon_3", charon_model)
]:
    results = model.val()  # uses model's saved val config & device
    mAP50_95 = results.box.map       # mAP@50:95
    AP50     = results.box.map50     # AP@50
    AP75     = results.box.map75     # AP@75
    ultra_metrics[name] = (mAP50_95, AP50, AP75)

###############################################################################
# Measure inference speed on /home/geuba03p/all_models_comparison/test_vid.mp4
cap = cv2.VideoCapture("/home/geuba03p/all_models_comparison/test_vid.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()
print(f"Loaded {len(frames)} frames from test video.")

def measure_speed(model, model_type="ultralytics"):
    # Warm-up for 2 frames
    if len(frames) < 2:
        print("[WARNING] Video has fewer than 2 frames, speed test might be inaccurate.")
    warmup_frames = min(2, len(frames))

    for i in range(warmup_frames):
        if model_type == "ultralytics":
            model.predict(frames[i], device=device, verbose=False)
        else:
            # TorchVision
            img = cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            tensor = torch.nn.functional.normalize(
                tensor, [0.485,0.456,0.406], [0.229,0.224,0.225]
            ).to(device)
            with torch.no_grad():
                model([tensor])

    # Sync GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()

    import time
    start = time.perf_counter()

    # Infer on each frame
    for frame in frames:
        if model_type == "ultralytics":
            _ = model.predict(frame, device=device, verbose=False)
        else:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img).permute(2,0,1).float() / 255.0
            tensor = torch.nn.functional.normalize(
                tensor, [0.485,0.456,0.406], [0.229,0.224,0.225]
            ).to(device)
            with torch.no_grad():
                _ = model([tensor])

        if device.type == 'cuda':
            torch.cuda.synchronize()

    total_time = time.perf_counter() - start
    avg_time_s = total_time / len(frames) if frames else 1.0
    avg_time_ms = avg_time_s * 1000.0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0
    return avg_time_ms, fps

speed_results = {}
speed_results["Faster R-CNN"] = measure_speed(fr_model, model_type="torchvision")
speed_results["FCOS"]         = measure_speed(fcos_model, model_type="torchvision")
speed_results["RetinaNet"]    = measure_speed(retina_model, model_type="torchvision")
speed_results["RT-DETR"]      = measure_speed(rtdetr_model, model_type="ultralytics")
speed_results["YOLOv8"]       = measure_speed(yolo8_model,  model_type="ultralytics")
speed_results["YOLO11"]       = measure_speed(yolo11_model, model_type="ultralytics")
speed_results["Charon_3"]     = measure_speed(charon_model, model_type="ultralytics")

###############################################################################
# Combine metrics and speed into one table
summary_data = []
all_model_names = ["Faster R-CNN", "FCOS", "RetinaNet", "RT-DETR", "YOLOv8", "YOLO11", "Charon_3"]
for model_name in all_model_names:
    # Grab the detection metrics (TorchVision in `metrics`, YOLO-based in `ultra_metrics`)
    # If in TorchVision dict, use that; if not, use ultra_metrics
    if model_name in metrics:
        (mAP50_95, AP50, AP75) = metrics[model_name]
    else:
        (mAP50_95, AP50, AP75) = ultra_metrics[model_name]

    # Speed info
    time_ms, fps = speed_results[model_name]
    summary_data.append({
        "Model": model_name,
        "mAP50-95": round(mAP50_95, 4),
        "AP50": round(AP50, 4),
        "AP75": round(AP75, 4),
        "InferenceTime(ms)": round(time_ms, 2),
        "FPS": round(fps, 2)
    })

df = pd.DataFrame(summary_data)

# Save summary CSV
output_dir = "/home/geuba03p/all_models_comparison/unified/evaluation"
os.makedirs(output_dir, exist_ok=True)
csv_path = os.path.join(output_dir, "all_models_summary.csv")
df.to_csv(csv_path, index=False)
print(f"Saved summary CSV to {csv_path}")
print(df.to_string(index=False))

###############################################################################
# Plot results
models = [row["Model"] for row in summary_data]
mAPs   = [row["mAP50-95"]*100 for row in summary_data]  # convert to %
AP50s  = [row["AP50"]*100 for row in summary_data]
AP75s  = [row["AP75"]*100 for row in summary_data]
times  = [row["InferenceTime(ms)"] for row in summary_data]
fps    = [row["FPS"] for row in summary_data]

# Accuracy bar chart
x = range(len(models))
width = 0.25
plt.figure(figsize=(8,5))
plt.bar([i - width for i in x], mAPs, width=width, label="mAP50-95")
plt.bar(x, AP50s, width=width, label="AP50")
plt.bar([i + width for i in x], AP75s, width=width, label="AP75")
plt.xticks(list(x), models, rotation=45, ha='right')
plt.ylabel("Average Precision (%)")
plt.title("Detection Accuracy per Model")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "metrics_comparison.png"))
plt.savefig(os.path.join(output_dir, "metrics_comparison.svg"))
plt.close()

# Inference time bar chart
plt.figure(figsize=(7,4))
plt.bar(models, times, color="skyblue")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Inference Time (ms/frame)")
plt.title("Inference Time per Frame (lower is better)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_time.png"))
plt.savefig(os.path.join(output_dir, "inference_time.svg"))
plt.close()

# FPS bar chart
plt.figure(figsize=(7,4))
plt.bar(models, fps, color="lightgreen")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Frames Per Second (FPS)")
plt.title("Inference Speed (higher is better)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_fps.png"))
plt.savefig(os.path.join(output_dir, "inference_fps.svg"))
plt.close()

print("[INFO] All done! Plots and CSV saved to", output_dir)
