"""
New Energy Model (ETA-style) with TensorBoard Logging
FIXED: Uses Cityscapes-trained Detectron2 model and correct label mapping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.modeling.poolers import ROIPooler
from detectron2.checkpoint import DetectionCheckpointer
import cv2, os, json, glob
from tqdm import tqdm
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# ===== CORRECT MAPPING: JSON category IDs (1-8) → Model training IDs (0-7) =====
JSON_TO_MODEL_ID = {
    1: 0,  # person → person
    2: 1,  # rider → rider
    3: 2,  # car → car
    4: 7,  # bicycle → bicycle (moves to end!)
    5: 6,  # motorcycle → motorcycle
    6: 4,  # bus → bus
    7: 3,  # truck → truck
    8: 5,  # train → train
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


# ===== FIXED: Setup Cityscapes-trained Detectron2 model =====
def setup_detectron2(device="cuda"):
    """
    Setup Detectron2 model trained on Cityscapes (NOT COCO!)
    """
    cfg = get_cfg()
    
    # Use COCO Faster R-CNN config as base architecture
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Override with Cityscapes-specific settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Cityscapes has 8 classes
    cfg.MODEL.WEIGHTS = "/home/lyz6/palmer_scratch/eta-object-detection/detectron2/tools/output/res50_fbn_1x/cityscapes_train_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = device
    
    # Build model
    model = build_model(cfg)
    model.to(device)
    
    # Load Cityscapes-trained weights
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    
    model.eval()
    
    print("✅ Loaded Cityscapes-trained Detectron2 model")
    print(f"   Model has {cfg.MODEL.ROI_HEADS.NUM_CLASSES} classes (Cityscapes)")
    
    return model, cfg


# ===== Load paired dataset (across all cities) =====
def load_paired_data(clean_dir, blur_dir, ann_file):
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # FIXED: Use correct mapping from JSON IDs to model training IDs
    img_id_to_anns = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        if ann["category_id"] in JSON_TO_MODEL_ID:
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            ann_copy = ann.copy()
            # Convert from JSON category ID (1-8) to model training ID (0-7)
            ann_copy["category_id"] = JSON_TO_MODEL_ID[ann["category_id"]]
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
        classes.append(ann["category_id"])  # Already remapped to 0-7
    gt.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32, device=device))
    gt.gt_classes = torch.tensor(classes, dtype=torch.int64, device=device)
    return gt


# ===== Compute per-detection error =====
def compute_detection_error(pred, gt):
    """L2-based unbounded error"""
    pred_boxes = pred.pred_boxes.tensor
    gt_boxes = gt.gt_boxes.tensor
    device = pred_boxes.device

    # Case 1: No predictions, no GT → no error needed (true negative)
    if len(pred_boxes) == 0 and len(gt_boxes) == 0:
        return torch.tensor([], device=device)
    
    # Case 2: No predictions, but GT exists → complete failure (missed detections)
    if len(pred_boxes) == 0:
        return torch.ones(len(gt_boxes), device=device) * 100.0
    
    # Case 3: Predictions exist, no GT → false positives (high error)
    if len(gt_boxes) == 0:
        return torch.ones(len(pred_boxes), device=device) * 100.0
    
    # Case 4: Both predictions and GT exist → compute matching error
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


# ===== Compute target energy (Gibbs transform) =====
def compute_target_energy(errors, temperature=0.5):
    return (1.0 - torch.exp(-errors / temperature)).clamp(0, 1)


# ===== Extract RoI features =====
def extract_roi_features(model, image_tensor, boxes):
    """
    Extract 7x7 ROI features from Detectron2 backbone using multi-level FPN.
    Uses the SAME multi-level pooling as the detector for proper alignment.
    """
    from detectron2.modeling.poolers import ROIPooler
    from detectron2.structures import Boxes
    
    # Extract FPN features
    features = model.backbone(image_tensor)
    
    # Use multi-level pooling matching the detector
    pooler = ROIPooler(
        output_size=7,
        scales=tuple(1.0 / model.backbone.output_shape()[k].stride 
                    for k in ["p2", "p3", "p4", "p5"]),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    )
    
    # Pool from ALL levels
    roi_boxes = [Boxes(boxes)]
    roi_feats = pooler(
        [features["p2"], features["p3"], features["p4"], features["p5"]],
        roi_boxes
    )
    
    return roi_feats


def run_inference(model, cfg, image):
    """
    Run inference using the Cityscapes-trained model.
    Returns Instances object with predictions.
    """
    device = next(model.parameters()).device
    height, width = image.shape[:2]
    
    # Preprocess image
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
    inputs = [{"image": image_tensor, "height": height, "width": width}]
    
    with torch.no_grad():
        images = model.preprocess_image(inputs)
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)
        results, _ = model.roi_heads(images, features, proposals, None)
    
    return results[0]


def train_energy_model(
    clean_dir,
    blur_dir,
    ann_file,
    num_epochs=10,
    learning_rate=1e-4,
    temperature=0.5,
    batch_accum=32,
    device="cuda",
    save_path="roi_energy_model.pth",
    log_dir="runs"
):
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"energy_model_T{temperature}_lr{learning_rate}_bs{batch_accum}_{timestamp}"
    writer = SummaryWriter(os.path.join(log_dir, experiment_name))
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', f"""
    - Learning Rate: {learning_rate}
    - Temperature: {temperature}
    - Batch Accumulation: {batch_accum}
    - Num Epochs: {num_epochs}
    - Device: {device}
    - Using Cityscapes-trained Detectron2 model
    """, 0)
    
    # FIXED: Setup Cityscapes-trained model
    model, cfg = setup_detectron2(device)
    energy_model = ROI_EnergyModel(in_channels=256).to(device)
    optimizer = optim.Adam(energy_model.parameters(), lr=learning_rate)

    # Load dataset
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)

    print(f"Training on {len(paired_data)} paired samples")
    writer.add_text('Dataset', f'Total samples: {len(paired_data)}', 0)

    global_step = 0
    
    for epoch in range(num_epochs):
        energy_model.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_loss = 0.0
        accum_count = 0
        
        # Track statistics per epoch
        epoch_pred_energy_vals = []
        epoch_target_energy_vals = []
        epoch_error_vals = []
        epoch_num_detections = []

        random.shuffle(paired_data)

        pbar = tqdm(paired_data, desc=f"Epoch {epoch+1}/{num_epochs}")

        for idx, data in enumerate(pbar):
            clean_img = cv2.imread(data["clean_path"])
            blur_img = cv2.imread(data["blur_path"])
            H, W = data["height"], data["width"]
            gt = prepare_gt_instances(data["annotations"], (H, W), device)

            # FIXED: Use our inference function
            with torch.no_grad():
                clean_out = run_inference(model, cfg, clean_img)
                blur_out = run_inference(model, cfg, blur_img)

            for pred_idx, (pred, img) in enumerate([(clean_out, clean_img), (blur_out, blur_img)]):
                if len(pred) == 0 or len(gt) == 0:
                    continue

                errors = compute_detection_error(pred, gt)
                target_E = compute_target_energy(errors, temperature)

                # Convert image to tensor and preprocess
                img_tensor = torch.as_tensor(img.astype("float32").transpose(2, 0, 1)).to(device)
                inputs = [{"image": img_tensor}]
                image_list = model.preprocess_image(inputs)

                # Extract ROI features
                roi_feats = extract_roi_features(model, image_list.tensor, pred.pred_boxes.tensor)

                # Forward pass
                pred_E_map = energy_model(roi_feats)
                target_E_map = target_E.view(-1, 1, 1, 1).expand_as(pred_E_map)

                # MSE loss
                loss = F.mse_loss(pred_E_map, target_E_map)

                # Accumulate gradients
                (loss / batch_accum).backward()
                batch_loss += loss.item()
                accum_count += 1
                epoch_loss += loss.item()
                num_batches += 1
                
                # Collect statistics
                epoch_pred_energy_vals.append(pred_E_map.detach().cpu())
                epoch_target_energy_vals.append(target_E_map.detach().cpu())
                epoch_error_vals.append(errors.detach().cpu())
                epoch_num_detections.append(len(pred))

                # Step optimizer every `batch_accum` samples
                if accum_count % batch_accum == 0:
                    pred_E_min = pred_E_map.min().item()
                    pred_E_max = pred_E_map.max().item()
                    pred_E_mean = pred_E_map.mean().item()
                    target_E_min = target_E_map.min().item()
                    target_E_max = target_E_map.max().item()
                    target_E_mean = target_E_map.mean().item()
                    
                    writer.add_scalar('Loss/batch', loss.item(), global_step)
                    writer.add_scalar('Energy/predicted_min', pred_E_min, global_step)
                    writer.add_scalar('Energy/predicted_max', pred_E_max, global_step)
                    writer.add_scalar('Energy/predicted_mean', pred_E_mean, global_step)
                    writer.add_scalar('Energy/target_min', target_E_min, global_step)
                    writer.add_scalar('Energy/target_max', target_E_max, global_step)
                    writer.add_scalar('Energy/target_mean', target_E_mean, global_step)
                    writer.add_scalar('Error/min', errors.min().item(), global_step)
                    writer.add_scalar('Error/max', errors.max().item(), global_step)
                    writer.add_scalar('Error/mean', errors.mean().item(), global_step)
                    writer.add_scalar('Detections/count', len(pred), global_step)
                    
                    global_step += 1
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    avg_batch_loss = batch_loss / batch_accum
                    pbar.set_postfix({
                        "avg_loss": f"{avg_batch_loss:.4f}", 
                        "N": len(pred),
                        "E_pred": f"[{pred_E_min:.3f},{pred_E_max:.3f}]",
                        "E_tgt": f"[{target_E_min:.3f},{target_E_max:.3f}]"
                    })
                    batch_loss = 0.0

        # Epoch summary
        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        all_pred_energy = torch.cat(epoch_pred_energy_vals)
        all_target_energy = torch.cat(epoch_target_energy_vals)
        all_errors = torch.cat(epoch_error_vals)
        
        writer.add_scalar('Loss/epoch', avg_epoch_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Summary/avg_pred_energy', all_pred_energy.mean().item(), epoch)
        writer.add_scalar('Summary/avg_target_energy', all_target_energy.mean().item(), epoch)
        writer.add_scalar('Summary/avg_error', all_errors.mean().item(), epoch)
        writer.add_scalar('Summary/avg_detections_per_image', np.mean(epoch_num_detections), epoch)
        
        energy_mse = F.mse_loss(all_pred_energy, all_target_energy).item()
        energy_mae = F.l1_loss(all_pred_energy, all_target_energy).item()
        writer.add_scalar('Metrics/energy_MSE', energy_mse, epoch)
        writer.add_scalar('Metrics/energy_MAE', energy_mae, epoch)
        
        print(f"\n✅ Epoch {epoch+1} complete — Avg Loss: {avg_epoch_loss:.4f}")
        print(f"   Energy MSE: {energy_mse:.4f}, MAE: {energy_mae:.4f}")
        print(f"   Pred Energy: [{all_pred_energy.min():.3f}, {all_pred_energy.max():.3f}], Mean: {all_pred_energy.mean():.3f}")
        print(f"   Target Energy: [{all_target_energy.min():.3f}, {all_target_energy.max():.3f}], Mean: {all_target_energy.mean():.3f}")

        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            ckpt_path = save_path.replace(".pth", f"_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': energy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_epoch_loss,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    torch.save({
        'model_state_dict': energy_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_epoch_loss,
        'hyperparameters': {
            'learning_rate': learning_rate,
            'temperature': temperature,
            'batch_accum': batch_accum,
            'num_epochs': num_epochs
        }
    }, save_path)
    
    print(f"\n🎯 Training complete! Model saved to {save_path}")
    print(f"📊 TensorBoard logs: {os.path.join(log_dir, experiment_name)}")
    
    writer.close()


# ===== Main =====
if __name__ == "__main__":
    CORRUPTION = "motion_blur"
    CLEAN_DIR = "/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/train"
    BLUR_DIR = f"/home/lyz6/AMROD/datasets/{CORRUPTION}/leftImg8bit/train"
    ANN_FILE = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_train.json"

    train_energy_model(
        clean_dir=CLEAN_DIR,
        blur_dir=BLUR_DIR,
        ann_file=ANN_FILE,
        num_epochs=2,
        learning_rate=5e-4,
        temperature=200,
        device="cuda",
        save_path=f"models/{CORRUPTION}_roi_energy_model.pth",
        log_dir="runs/energy_model"
    )