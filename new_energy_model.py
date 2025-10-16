"""
Energy-Based Test-Time Adaptation for Object Detection (ETA-style)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.poolers import ROIPooler
import cv2, os, json, glob
from tqdm import tqdm
import numpy as np
import random

# ===== Cityscapes → COCO class mapping =====
CITYSCAPES_TO_COCO = {
    1: 0,  # person
    2: 0,  # rider → person
    3: 2,  # car
    4: 1,  # bicycle
    5: 3,  # motorcycle
    6: 5,  # bus
    7: 7,  # truck
    8: 6,  # train
}


# ===== ETA-style Region Energy Model =====
class ROI_EnergyModel(nn.Module):
    """
    CNN-based ETA-style energy model for object detection.
    Takes RoI-aligned features and outputs a spatial energy map per RoI.
    """

    def __init__(self, in_channels=256):
        super().__init__()

        channels = [in_channels, 64, 128, 256, 256, 512, 512]
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i + 1], 5, stride=2, padding=2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.conv_layers = nn.Sequential(*layers)

        # Final 3×3 conv → single-channel energy map
        self.final_conv = nn.Conv2d(512, 1, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, roi_feats):
        """
        roi_feats: (N, C, H, W)
        returns: energy_maps (N, 1, H', W')
        """
        x = self.conv_layers(roi_feats)
        return self.sigmoid(self.final_conv(x))

def energy_loss(pred_energy, target_energy):
    eps = 1e-8
    pred = pred_energy.view(pred_energy.size(0), -1)
    target = target_energy.view(target_energy.size(0), -1)
    pred = pred / (pred.sum(dim=1, keepdim=True) + eps)
    target = target / (target.sum(dim=1, keepdim=True) + eps)
    loss = torch.sum(target * torch.log((target + eps) / (pred + eps)), dim=1)
    return torch.mean(loss)


# ===== Detectron2 Setup =====
def setup_detectron2(device="cuda"):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = device
    return DefaultPredictor(cfg)


# ===== Load paired dataset (across all cities) =====
def load_paired_data(clean_dir, blur_dir, ann_file):
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if ann["category_id"] in CITYSCAPES_TO_COCO:
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            ann_copy = ann.copy()
            ann_copy["category_id"] = CITYSCAPES_TO_COCO[ann["category_id"]]
            img_id_to_anns[img_id].append(ann_copy)

    paired_data = []
    for img_info in coco_data["images"]:
        file_name = img_info["file_name"].split("/")[-1]
        # Search across all subfolders
        clean_path = None
        blur_path = None
        for root, _, files in os.walk(clean_dir):
            if file_name in files:
                clean_path = os.path.join(root, file_name)
                break
        for root, _, files in os.walk(blur_dir):
            if file_name in files:
                blur_path = os.path.join(root, file_name)
                break

        if not (clean_path and blur_path):
            continue

        img_id = img_info["id"]
        anns = img_id_to_anns.get(img_id, [])
        if len(anns) > 0:
            paired_data.append({
                "clean_path": clean_path,
                "blur_path": blur_path,
                "height": img_info["height"],
                "width": img_info["width"],
                "annotations": anns,
                "image_id": img_id
            })
    print(f"Total paired samples: {len(paired_data)}")
    return paired_data


# ===== Prepare GT instances =====
def prepare_gt_instances(annotations, image_size, device):
    gt = Instances(image_size)
    if len(annotations) == 0:
        gt.gt_boxes = Boxes(torch.zeros((0, 4), device=device))
        gt.gt_classes = torch.zeros((0,), dtype=torch.int64, device=device)
        return gt
    boxes = []
    classes = []
    for ann in annotations:
        x, y, w, h = ann["bbox"]
        boxes.append([x, y, x + w, y + h])
        classes.append(ann["category_id"])
    gt.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32, device=device))
    gt.gt_classes = torch.tensor(classes, dtype=torch.int64, device=device)
    return gt


# ===== Compute per-detection error =====

# Problem with IOU -- it is bounded between [-1, 1]
def compute_detection_error(pred, gt):
    """L2-based unbounded error"""
    pred_boxes = pred.pred_boxes.tensor
    gt_boxes = gt.gt_boxes.tensor
    device = pred_boxes.device

    if len(pred_boxes) == 0:
        return torch.tensor([], device=device)
    if len(gt_boxes) == 0:
        return torch.ones(len(pred_boxes), device=device) * 100.0

    # Centers and sizes
    pred_centers = (pred_boxes[:, :2] + pred_boxes[:, 2:]) / 2.0
    pred_sizes = pred_boxes[:, 2:] - pred_boxes[:, :2]
    
    gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2.0
    gt_sizes = gt_boxes[:, 2:] - gt_boxes[:, :2]
    
    # Pairwise L2 distances
    center_dist = torch.cdist(pred_centers, gt_centers, p=2)
    pred_sizes_exp = pred_sizes.unsqueeze(1)
    gt_sizes_exp = gt_sizes.unsqueeze(0)
    size_dist = torch.norm(pred_sizes_exp - gt_sizes_exp, dim=2)
    
    # Combined error
    total_error = center_dist + 0.5 * size_dist
    
    # Match to closest GT
    best_error, _ = total_error.min(dim=1)
    
    return best_error

"""
def compute_detection_error(pred, gt):
    Compute pure IoU-based error per detection (analog of MSE for depth).
    Error = 1 - IoU (so perfect detection has error=0, no overlap has error=1)
    pred_boxes = pred.pred_boxes
    gt_boxes = gt.gt_boxes
    device = pred_boxes.device

    if len(pred_boxes) == 0:
        return torch.tensor([], device=device)
    if len(gt_boxes) == 0:
        # No GT means all predictions are false positives → max error
        return torch.ones(len(pred_boxes), device=device)

    # Compute IoU matrix: (N_pred, N_gt)
    iou = pairwise_iou(pred_boxes, gt_boxes)
    
    # For each prediction, find best matching GT box
    best_iou, best_idx = iou.max(dim=1)
    
    # Error = 1 - IoU (pure geometric error)
    error = 1.0 - best_iou
    
    return error
"""

# ===== Compute target energy (Gibbs transform) =====
def compute_target_energy(errors, temperature=0.5):
    return (1.0 - torch.exp(-errors / temperature)).clamp(0, 1)

# ===== Extract RoI features =====
def extract_roi_features(model, image_tensor, boxes):
    """
    Extract 7x7 ROI features from Detectron2 backbone.
    """
    features = model.backbone(image_tensor)
    pooler = ROIPooler(
        output_size=7,
        scales=(1.0 / model.backbone.output_shape()["p2"].stride,),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    )
    roi_boxes = [Boxes(boxes)]
    roi_feats = pooler([features["p2"]], roi_boxes)
    return roi_feats


def train_energy_model(
    clean_dir,
    blur_dir,
    ann_file,
    num_epochs=10,
    learning_rate=1e-4,
    temperature=0.5,
    batch_accum=32,        # <-- gradient accumulation steps
    device="cuda",
    save_path="roi_energy_model.pth"
):
    # Setup models
    predictor = setup_detectron2(device)
    energy_model = ROI_EnergyModel(in_channels=256).to(device)
    optimizer = optim.Adam(energy_model.parameters(), lr=learning_rate)

    # Load dataset
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)
    relevant_classes = [0, 1, 2, 3, 5, 6, 7]

    print(f"Training on {len(paired_data)} paired samples")

    for epoch in range(num_epochs):
        energy_model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_loss = 0.0
        accum_count = 0

        random.shuffle(paired_data)

        pbar = tqdm(paired_data, desc=f"Epoch {epoch+1}/{num_epochs}")

        for idx, data in enumerate(pbar):
            clean_img = cv2.imread(data["clean_path"])
            blur_img = cv2.imread(data["blur_path"])
            H, W = data["height"], data["width"]
            gt = prepare_gt_instances(data["annotations"], (H, W), device)

            with torch.no_grad():
                clean_out = predictor(clean_img)["instances"].to(device)
                blur_out = predictor(blur_img)["instances"].to(device)

            for pred in [clean_out, blur_out]:
                if len(pred) == 0 or len(gt) == 0:
                    continue

                errors = compute_detection_error(pred, gt)
                
                """
                print(f"Error stats - Min: {errors.min():.4f}, Max: {errors.max():.4f}, "
                    f"Mean: {errors.mean():.4f}, Median: {errors.median():.4f}")
                """

                target_E = compute_target_energy(errors, temperature)


                # Convert image to tensor and preprocess
                img_tensor = torch.as_tensor(clean_img.astype("float32").transpose(2, 0, 1)).to(device)
                inputs = [{"image": img_tensor}]
                image_list = predictor.model.preprocess_image(inputs)

                # Extract ROI features
                roi_feats = extract_roi_features(predictor.model, image_list.tensor, pred.pred_boxes.tensor)

                # Forward pass
                pred_E_map = energy_model(roi_feats)  # (N, 1, h, w)
                target_E_map = target_E.view(-1, 1, 1, 1).expand_as(pred_E_map)

                # Binary cross-entropy loss
                # loss = F.binary_cross_entropy(pred_E_map, target_E_map)
                loss = F.mse_loss(pred_E_map, target_E_map)

                # Accumulate gradients
                (loss / batch_accum).backward()
                batch_loss += loss.item()
                accum_count += 1
                epoch_loss += loss.item()
                num_batches += 1

                if accum_count % batch_accum == 0:
                    print(f"Pred E range: [{pred_E_map.min():.3f}, {pred_E_map.max():.3f}], "
                        f"Target E range: [{target_E_map.min():.3f}, {target_E_map.max():.3f}]")


                # Step optimizer every `batch_accum` samples
                if accum_count % batch_accum == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    avg_batch_loss = batch_loss / batch_accum
                    pbar.set_postfix({"avg_loss": f"{avg_batch_loss:.4f}", "N": len(pred)})
                    batch_loss = 0.0

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"\n✅ Epoch {epoch+1} complete — Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_path = save_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save(energy_model.state_dict(), ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    torch.save(energy_model.state_dict(), save_path)
    print(f"\n🎯 Training complete! Model saved to {save_path}")

# ===== Main =====
if __name__ == "__main__":
    CLEAN_DIR = "/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/val"
    BLUR_DIR = "/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val"
    ANN_FILE = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"

    train_energy_model(
        clean_dir=CLEAN_DIR,
        blur_dir=BLUR_DIR,
        ann_file=ANN_FILE,
        num_epochs=3,
        learning_rate=5e-4,
        temperature=200,
        device="cuda",
        save_path="roi_energy_model.pth"
    )
