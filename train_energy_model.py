"""
Train Energy Model for Object Detection - COMPLETE FIXED VERSION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances, pairwise_iou
import cv2
import json
import os
from tqdm import tqdm
import numpy as np


# Cityscapes (1-8) to COCO class mapping
CITYSCAPES_TO_COCO = {
    1: 0,   # person -> person
    2: 0,   # rider -> person
    3: 2,   # car -> car
    4: 1,   # bicycle -> bicycle
    5: 3,   # motorcycle -> motorcycle
    6: 5,   # bus -> bus
    7: 7,   # truck -> truck
    8: 6,   # train -> train
}

class DetectionEnergyModel(nn.Module):
    """Energy model that scores detection reliability"""
    
    def __init__(self, box_dim=4, num_classes=80, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        input_dim = box_dim + num_classes + 1
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, boxes, class_probs, scores):
        x = torch.cat([boxes, class_probs, scores.unsqueeze(-1)], dim=-1)
        return self.network(x)


def setup_detectron2(device='cuda'):
    """Setup Detectron2 predictor"""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.DEVICE = device
    
    return DefaultPredictor(cfg)


def load_paired_data(clean_dir, blur_dir, ann_file):
    """Load paired clean/blur images with annotations"""

    
    with open(ann_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build mappings
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if ann['category_id'] in CITYSCAPES_TO_COCO:
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            ann_copy = ann.copy()
            ann_copy['category_id'] = CITYSCAPES_TO_COCO[ann['category_id']]
            img_id_to_anns[img_id].append(ann_copy)
    
    print(f"Total annotations after remapping: {sum(len(v) for v in img_id_to_anns.values())}")
    
    # Create paired dataset
    paired_data = []
    not_found_count = 0
    
    for img_info in coco_data['images']:
        full_filename = img_info['file_name']
        
        # Extract city and filename
        parts = full_filename.split('/')
        if len(parts) >= 3:
            city = parts[2]
            filename = parts[-1]
        else:
            filename = full_filename.split('/')[-1]
            city = None
        
        clean_path = os.path.join(clean_dir, city, filename)
        blur_path  = os.path.join(blur_dir, city, filename)

        if not (os.path.exists(clean_path) and os.path.exists(blur_path)):
            not_found_count += 1
            if not_found_count <= 3:  # Print first few missing files
                print(f"Missing: {filename}")
            continue
        
        img_id = img_info['id']
        annotations = img_id_to_anns.get(img_id, [])
        
        if len(annotations) > 0:
            paired_data.append({
                'clean_path': clean_path,
                'blur_path': blur_path,
                'height': img_info['height'],
                'width': img_info['width'],
                'annotations': annotations,
                'image_id': img_id
            })
    
    print(f"Files not found: {not_found_count}")
    print(f"Successfully paired images: {len(paired_data)}")
    
    return paired_data


def prepare_gt_instances(annotations, image_size, device):
    """Convert COCO annotations to Detectron2 Instances"""
    gt_instances = Instances(image_size)
    
    if len(annotations) == 0:
        gt_instances.gt_boxes = Boxes(torch.zeros((0, 4), device=device))
        gt_instances.gt_classes = torch.zeros((0,), dtype=torch.int64, device=device)
        return gt_instances
    
    boxes = []
    classes = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])
        classes.append(ann['category_id'])
    
    gt_instances.gt_boxes = Boxes(torch.tensor(boxes, dtype=torch.float32, device=device))
    gt_instances.gt_classes = torch.tensor(classes, dtype=torch.int64, device=device)
    
    return gt_instances


def compute_detection_error(pred_instances, gt_instances, w_box=1.0, w_cls=0.5, w_conf=0.3):
    """Compute per-detection error"""
    pred_boxes = pred_instances.pred_boxes
    pred_classes = pred_instances.pred_classes
    pred_scores = pred_instances.scores
    
    gt_boxes = gt_instances.gt_boxes
    gt_classes = gt_instances.gt_classes
    
    device = pred_scores.device
    
    if len(pred_boxes) == 0:
        return torch.tensor([], device=device)
    
    if len(gt_boxes) == 0:
        return torch.ones(len(pred_boxes), device=device)
    
    iou_matrix = pairwise_iou(pred_boxes, gt_boxes)
    matched_iou, matched_gt_idx = iou_matrix.max(dim=1)
    
    box_error = 1.0 - matched_iou
    
    gt_classes_matched = gt_classes[matched_gt_idx]
    cls_error = (pred_classes != gt_classes_matched).float()
    
    conf_error = 1.0 - pred_scores
    
    total_error = w_box * box_error + w_cls * cls_error + w_conf * conf_error
    
    total_error[matched_iou < 0.5] = 1.0
    
    return total_error


def compute_target_energy(errors, temperature=0.5):
    """Map errors to target energy using Gibbs distribution"""
    return (1.0 - torch.exp(-errors / temperature)).clamp(0, 1)


def normalize_boxes(boxes_tensor, image_size):
    """Normalize box coordinates to [0, 1]"""
    height, width = image_size
    boxes_norm = boxes_tensor.clone()
    boxes_norm[:, [0, 2]] /= width
    boxes_norm[:, [1, 3]] /= height
    return boxes_norm.clamp(0, 1)


def prepare_energy_inputs(pred_instances, image_size, num_classes=80):
    """Prepare inputs for energy model"""
    if len(pred_instances) == 0:
        device = pred_instances.pred_boxes.device
        return (
            torch.zeros((0, 4), device=device),
            torch.zeros((0, num_classes), device=device),
            torch.zeros((0,), device=device)
        )
    
    boxes_norm = normalize_boxes(pred_instances.pred_boxes.tensor, image_size)
    pred_classes = pred_instances.pred_classes
    class_probs = F.one_hot(pred_classes, num_classes).float()
    scores = pred_instances.scores
    
    return boxes_norm, class_probs, scores


def filter_relevant_predictions(pred_instances, relevant_classes=[0, 1, 2, 3, 5, 6, 7]):
    """Filter predictions to relevant COCO classes"""
    if len(pred_instances) == 0:
        return pred_instances
    
    mask = torch.zeros(len(pred_instances), dtype=torch.bool, device=pred_instances.pred_classes.device)
    for cls in relevant_classes:
        mask |= (pred_instances.pred_classes == cls)
    
    return pred_instances[mask]


def train_energy_model(
    clean_dir,
    blur_dir,
    ann_file,
    num_epochs=10,
    learning_rate=1e-4,
    temperature=0.5,
    num_classes=80,
    device='cuda',
    save_path='energy_model.pth'
):
    """Main training function"""
    print("Setting up...")
    
    predictor = setup_detectron2(device=device)
    energy_model = DetectionEnergyModel(num_classes=num_classes).to(device)
    optimizer = optim.Adam(energy_model.parameters(), lr=learning_rate)
    
    print("Loading paired dataset...")
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)
    
    if len(paired_data) == 0:
        print("ERROR: No paired data found!")
        return
    
    relevant_classes = [0, 1, 2, 3, 5, 6, 7]
    
    for epoch in range(num_epochs):
        energy_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        total_clean_dets = 0
        total_blur_dets = 0
        total_gt_boxes = 0
        num_nonzero_loss = 0
        
        pbar = tqdm(paired_data, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for data in pbar:
            clean_img = cv2.imread(data['clean_path'])
            blur_img = cv2.imread(data['blur_path'])
            image_size = (data['height'], data['width'])
            
            gt_instances = prepare_gt_instances(data['annotations'], image_size, device)
            total_gt_boxes += len(gt_instances)
            
            with torch.no_grad():
                clean_outputs = predictor(clean_img)
                clean_instances = filter_relevant_predictions(
                    clean_outputs['instances'].to(device), relevant_classes
                )
                
                blur_outputs = predictor(blur_img)
                blur_instances = filter_relevant_predictions(
                    blur_outputs['instances'].to(device), relevant_classes
                )
            
            total_clean_dets += len(clean_instances)
            total_blur_dets += len(blur_instances)
            
            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_samples = 0
            
            if len(clean_instances) > 0 and len(gt_instances) > 0:
                clean_errors = compute_detection_error(clean_instances, gt_instances)
                
                if len(clean_errors) > 0:
                    clean_target_energy = compute_target_energy(clean_errors, temperature)
                    boxes_norm, class_probs, scores = prepare_energy_inputs(
                        clean_instances, image_size, num_classes
                    )
                    clean_pred_energy = energy_model(boxes_norm, class_probs, scores).squeeze(-1)
                    loss_clean = F.binary_cross_entropy(clean_pred_energy, clean_target_energy)
                    total_loss = total_loss + loss_clean
                    n_samples += 1
            
            if len(blur_instances) > 0 and len(gt_instances) > 0:
                blur_errors = compute_detection_error(blur_instances, gt_instances)
                
                if len(blur_errors) > 0:
                    blur_target_energy = compute_target_energy(blur_errors, temperature)
                    boxes_norm, class_probs, scores = prepare_energy_inputs(
                        blur_instances, image_size, num_classes
                    )
                    blur_pred_energy = energy_model(boxes_norm, class_probs, scores).squeeze(-1)
                    loss_blur = F.binary_cross_entropy(blur_pred_energy, blur_target_energy)
                    total_loss = total_loss + loss_blur
                    n_samples += 1
            
            if n_samples > 0 and total_loss.item() > 0:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
                num_batches += 1
                num_nonzero_loss += 1
                
                pbar.set_postfix({
                    'loss': f'{total_loss.item():.4f}',
                    'clean': len(clean_instances),
                    'blur': len(blur_instances)
                })
        
        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Total GT boxes: {total_gt_boxes}")
        print(f"  Total clean detections: {total_clean_dets}")
        print(f"  Total blur detections: {total_blur_dets}")
        print(f"  Batches with nonzero loss: {num_nonzero_loss}/{len(paired_data)}")
        
        if (epoch + 1) % 2 == 0:
            checkpoint_path = save_path.replace('.pth', f'_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': energy_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)
            print(f"Saved: {checkpoint_path}")
    
    torch.save({
        'model_state_dict': energy_model.state_dict(),
        'config': {'num_classes': num_classes, 'temperature': temperature}
    }, save_path)
    print(f"Training complete! Saved to {save_path}")


if __name__ == "__main__":
    CLEAN_DIR = "/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/val"
    BLUR_DIR = "/home/lyz6/AMROD/datasets/snow/leftImg8bit/val"
    ANN_FILE = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
    
    train_energy_model(
        clean_dir=CLEAN_DIR,
        blur_dir=BLUR_DIR,
        ann_file=ANN_FILE,
        num_epochs=10,
        learning_rate=1e-4,
        temperature=0.5,
        num_classes=80,
        device='cuda',
        save_path='energy_model.pth'
    )