"""
Test-Time Adaptation with Batch Normalization and ROI-based Energy Guidance
Combines BN adaptation with ROI-based energy model guidance for object detection

This script uses an energy model that:
1. Takes ROI-aligned features from detections
2. Predicts energy maps per detection
3. Guides adaptation by minimizing predicted energy

The energy model expects:
- Input: (N, 256, 7, 7) ROI-aligned features per detection
- Output: (N, 1, h, w) energy maps per detection
"""

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Instances, Boxes
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers import FrozenBatchNorm2d
from detectron2.modeling.poolers import ROIPooler

from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')


# ==============================================================
# CLASS MAPPING
# ==============================================================


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

# ==============================================================
# BN CONVERSION UTILITIES
# ==============================================================

def convert_frozen_batchnorm_to_batchnorm(module):
    """
    Recursively convert all FrozenBatchNorm2d layers to BatchNorm2d.
    This allows the BN statistics to be updated during test-time adaptation.
    """
    module_output = module
    if isinstance(module, FrozenBatchNorm2d):
        # Create a new BatchNorm2d layer
        bn = nn.BatchNorm2d(
            module.num_features,
            eps=module.eps,
            affine=True,
            track_running_stats=True
        )
        
        # Copy weights and biases
        bn.weight.data = module.weight.data.clone()
        bn.bias.data = module.bias.data.clone()
        
        # Copy running statistics if they exist
        if hasattr(module, 'running_mean'):
            bn.running_mean.data = module.running_mean.data.clone()
        if hasattr(module, 'running_var'):
            bn.running_var.data = module.running_var.data.clone()
        
        module_output = bn
    
    # Recursively apply to child modules
    for name, child in module.named_children():
        module_output.add_module(name, convert_frozen_batchnorm_to_batchnorm(child))
    
    del module
    return module_output


def save_bn_state(model):
    """Save the current state of all BN layers."""
    bn_state = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            bn_state[name] = {
                'weight': module.weight.data.clone() if module.weight is not None else None,
                'bias': module.bias.data.clone() if module.bias is not None else None,
                'running_mean': module.running_mean.clone(),
                'running_var': module.running_var.clone(),
                'num_batches_tracked': module.num_batches_tracked.clone()
            }
    return bn_state


def restore_bn_state(model, bn_state):
    """Restore BN layers to a saved state."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)) and name in bn_state:
            state = bn_state[name]
            if state['weight'] is not None:
                module.weight.data.copy_(state['weight'])
            if state['bias'] is not None:
                module.bias.data.copy_(state['bias'])
            module.running_mean.copy_(state['running_mean'])
            module.running_var.copy_(state['running_var'])
            module.num_batches_tracked.copy_(state['num_batches_tracked'])


def reset_bn_stats(model):
    """Reset BN running statistics to initial values."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.track_running_stats:
                module.running_mean.zero_()
                module.running_var.fill_(1)
                module.num_batches_tracked.zero_()


def set_bn_to_train(model):
    """Set all BN layers to training mode."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()


def set_bn_to_eval(model):
    """Set all BN layers to eval mode."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()


# ==============================================================
# DETECTRON2 SETUP WITH BN ADAPTATION
# ==============================================================

def setup_detectron2_for_bn_adaptation(device="cuda", checkpoint_path=None):
    """
    Setup Detectron2 model with BN layers converted for adaptation.
    
    Args:
        device: Device to load model on
        checkpoint_path: Optional path to load weights from a previous checkpoint.
                        If None, loads COCO pretrained weights.
    """
    cfg = get_cfg()

    # Use COCO Faster R-CNN config as base (NOT Mask R-CNN)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Override with Cityscapes-specific settings
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # Cityscapes has 8 classes
    cfg.MODEL.WEIGHTS = "/home/lyz6/palmer_scratch/eta-object-detection/detectron2/tools/output/res50_fbn_1x/cityscapes_train_final.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    # Build model
    model = build_model(cfg)
    model = model.to(device)
    
    # Load weights
    checkpointer = DetectionCheckpointer(model)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading weights from custom checkpoint: {checkpoint_path}")
        checkpointer.load(checkpoint_path)
    else:
        print("   Loading COCO pretrained weights...")
        checkpointer.load(cfg.MODEL.WEIGHTS)
    
    print(f"✅ Loaded weights successfully")
    
    # Count BN layers before conversion
    frozen_bn_count = sum(1 for m in model.modules() if isinstance(m, FrozenBatchNorm2d))
    print(f"   FrozenBatchNorm2d layers before conversion: {frozen_bn_count}")
    
    # Convert FrozenBatchNorm2d to BatchNorm2d
    print("   Converting FrozenBatchNorm2d to BatchNorm2d...")
    model = convert_frozen_batchnorm_to_batchnorm(model)
    
    # Count BN layers after conversion
    bn_count = sum(1 for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)))
    frozen_bn_remaining = sum(1 for m in model.modules() if isinstance(m, FrozenBatchNorm2d))
    print(f"   BatchNorm2d layers after conversion: {bn_count}")
    print(f"   FrozenBatchNorm2d remaining: {frozen_bn_remaining}")
    
    # Freeze all parameters except BN affine parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Make BN affine parameters trainable
    bn_params_trainable = 0
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.weight is not None:
                module.weight.requires_grad = True
                bn_params_trainable += module.weight.numel()
            if module.bias is not None:
                module.bias.requires_grad = True
                bn_params_trainable += module.bias.numel()
    
    print(f"   Trainable BN affine parameters: {bn_params_trainable:,}")
    
    return model, cfg


# ==============================================================
# DATASET LOADING
# ==============================================================

def load_cityscapes_data(image_dir, annotation_file, max_images=None):
    """Load Cityscapes validation data."""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Build image_id to annotations mapping
    img_id_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        # Remap from JSON category ID to model training ID
        if ann['category_id'] in JSON_TO_MODEL_ID:
            ann_copy = ann.copy()
            ann_copy['category_id'] = JSON_TO_MODEL_ID[ann['category_id']]
            img_id_to_anns[ann['image_id']].append(ann_copy)
    
    # Build dataset
    dataset = []
    for img_info in coco_data['images']:
        file_name = img_info['file_name']
        
        # Handle different directory structures
        if '/' in file_name:
            file_name = file_name.split('/')[-1]
        
        # Search for image using os.walk
        image_path = None
        for root, dirs, files in os.walk(image_dir):
            if file_name in files:
                image_path = os.path.join(root, file_name)
                break
        
        if image_path is None:
            continue
        
        img_id = img_info['id']
        anns = img_id_to_anns.get(img_id, [])
        
        dataset.append({
            'image_path': image_path,
            'annotations': anns,
            'height': img_info['height'],
            'width': img_info['width'],
            'image_id': img_id
        })
        
        if max_images and len(dataset) >= max_images:
            break
    
    print(f"✅ Loaded {len(dataset)} images with annotations")
    return dataset


def prepare_gt_instances(annotations, image_size, device):
    """Convert annotations to GT dict"""
    if len(annotations) == 0:
        return {
            'boxes': torch.zeros((0, 4), device=device),
            'labels': torch.zeros((0,), dtype=torch.int64, device=device)
        }
    
    boxes = []
    labels = []
    for ann in annotations:
        x, y, w, h = ann['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(ann['category_id'])  # Already mapped to COCO
    
    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32, device=device),
        'labels': torch.tensor(labels, dtype=torch.int64, device=device)
    }


# ==============================================================
# PREDICTION FILTERING
# ==============================================================

def filter_and_remap_predictions(predictions, score_threshold=0.05):
    """Filter predictions by score threshold only."""
    if not isinstance(predictions, Instances):
        return predictions
    
    pred_boxes = predictions.pred_boxes.tensor
    pred_classes = predictions.pred_classes
    pred_scores = predictions.scores
    
    # Filter by score only
    keep_mask = pred_scores > score_threshold
    
    return {
        'boxes': pred_boxes[keep_mask],
        'labels': pred_classes[keep_mask],
        'scores': pred_scores[keep_mask]
    }


# ==============================================================
# mAP COMPUTATION
# ==============================================================

def compute_iou(box1, box2):
    """Compute IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0


def compute_average_precision(precisions, recalls):
    """Compute AP from precision-recall curve"""
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0] + 1
    ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
    
    return ap


def compute_map(predictions_list, ground_truths, iou_threshold=0.5, verbose=False):
    """Compute mAP across multiple images."""
    all_predictions = defaultdict(list)
    all_ground_truths = defaultdict(list)
    
    for img_idx, (pred, gt) in enumerate(zip(predictions_list, ground_truths)):
        # Ground truths
        gt_boxes = gt['boxes'].cpu().numpy()
        gt_labels = gt['labels'].cpu().numpy()
        
        for label in np.unique(gt_labels):
            mask = gt_labels == label
            all_ground_truths[int(label)].append({
                'boxes': gt_boxes[mask],
                'img_idx': img_idx,
                'detected': np.zeros(np.sum(mask), dtype=bool)
            })
        
        # Predictions
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()
        
        for i, (box, label, score) in enumerate(zip(pred_boxes, pred_labels, pred_scores)):
            all_predictions[int(label)].append({
                'box': box,
                'score': score,
                'img_idx': img_idx
            })
    
    # Compute AP for each class
    aps = []
    for class_id in range(8):  # Cityscapes has classes 0-7
        if class_id not in all_predictions or class_id not in all_ground_truths:
            continue
        
        # Sort predictions by score (descending)
        preds = all_predictions[class_id]
        preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Count total ground truths for this class
        num_gts = sum(len(gt['boxes']) for gt in all_ground_truths[class_id])
        
        if num_gts == 0:
            continue
        
        # Match predictions to ground truths
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        for pred_idx, pred in enumerate(preds):
            img_idx = pred['img_idx']
            pred_box = pred['box']
            
            # Find GTs for this image
            gt_data = None
            for gt in all_ground_truths[class_id]:
                if gt['img_idx'] == img_idx:
                    gt_data = gt
                    break
            
            if gt_data is None or len(gt_data['boxes']) == 0:
                fp[pred_idx] = 1
                continue
            
            # Compute IoU with all GTs
            ious = np.array([compute_iou(pred_box, gt_box) for gt_box in gt_data['boxes']])
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            # Check if match
            if max_iou >= iou_threshold and not gt_data['detected'][max_iou_idx]:
                tp[pred_idx] = 1
                gt_data['detected'][max_iou_idx] = True
            else:
                fp[pred_idx] = 1
        
        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # Compute AP
        ap = compute_average_precision(precisions, recalls)
        aps.append(ap)
        
        if verbose:
            print(f"    Class {class_id}: AP = {ap:.3f} (GTs: {num_gts}, Preds: {len(preds)})")
    
    # Return mean AP
    return np.mean(aps) if len(aps) > 0 else 0.0


# ==============================================================
# ENERGY MODEL UTILITIES
# ==============================================================

def extract_roi_features(model, image_tensor, boxes):
    """
    Extract 7x7 ROI features from Detectron2 backbone using multi-level FPN.
    This matches the extraction method used during energy model training.
    
    Args:
        model: Detectron2 model with FPN backbone
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        boxes: Tensor of boxes [N, 4] in format [x1, y1, x2, y2]
    
    Returns:
        roi_feats: [N, 256, 7, 7] ROI-aligned features
    """
    # Extract FPN features
    features = model.backbone(image_tensor)
    
    # Use multi-level pooling matching the detector
    pooler = ROIPooler(
        output_size=7,
        # Include ALL pyramid levels that the detector uses
        scales=tuple(1.0 / model.backbone.output_shape()[k].stride 
                    for k in ["p2", "p3", "p4", "p5"]),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    )
    
    # Pool from ALL levels - ROIPooler automatically selects correct level per box
    roi_boxes = [Boxes(boxes)]
    roi_feats = pooler(
        [features["p2"], features["p3"], features["p4"], features["p5"]],
        roi_boxes
    )
    
    return roi_feats


def compute_energy_loss(model, energy_model, image_tensor, height, width, device, score_threshold=0.05):
    """
    Compute energy loss for the current predictions using ROI-aligned features.
    
    This implementation:
    1. Runs detection to get predicted boxes
    2. Extracts ROI-aligned features for each detection
    3. Passes features through energy model
    4. Returns mean energy as loss
    
    Args:
        model: Detection model
        energy_model: Trained ROI_EnergyModel
        image_tensor: Input image tensor
        height: Image height
        width: Image width
        device: Device
        score_threshold: Minimum score for detections
    
    Returns:
        energy_loss: Mean energy across all detections
    """
    # Preprocess image
    inputs = [{"image": image_tensor, "height": height, "width": width}]
    images = model.preprocess_image(inputs)
    
    # Get predictions
    with torch.no_grad():
        features = model.backbone(images.tensor)
        proposals, _ = model.proposal_generator(images, features, None)
        results, _ = model.roi_heads(images, features, proposals, None)
        predictions = results[0]
    
    # Filter predictions by score
    keep_mask = predictions.scores > score_threshold
    
    if keep_mask.sum() == 0:
        # No detections, return large loss
        return torch.tensor(1, device=device, requires_grad=True)
    
    # Filter to keep only confident predictions
    pred_boxes = predictions.pred_boxes.tensor[keep_mask]
    
    # Extract ROI features for these boxes (requires grad for backprop)
    # We need to re-compute features with gradients enabled
    features = model.backbone(images.tensor)
    roi_feats = extract_roi_features(model, images.tensor, pred_boxes)
    
    # Compute energy for each ROI
    energy_maps = energy_model(roi_feats)  # (N, 1, h, w)
    
    # Mean energy across all detections and spatial locations
    energy_loss = energy_maps.mean()
    
    return energy_loss

"""
Visualization Script for Test-Time Adaptation
Outputs two images: before and after ETA with bounding boxes and labels
"""

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add the visualization function
def visualize_predictions(image, predictions, class_names, title="Detections", score_threshold=0.5):
    """
    Visualize predictions on image with bounding boxes and labels.
    
    Args:
        image: numpy array (H, W, 3) in BGR format
        predictions: dict with 'boxes', 'labels', 'scores'
        class_names: dict mapping class IDs to names
        title: title for the plot
        score_threshold: minimum confidence to display
    
    Returns:
        fig: matplotlib figure
    """
    # Convert BGR to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Filter by confidence
    keep_mask = predictions['scores'] >= score_threshold
    boxes = predictions['boxes'][keep_mask].cpu().numpy()
    labels = predictions['labels'][keep_mask].cpu().numpy()
    scores = predictions['scores'][keep_mask].cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image_rgb)
    ax.set_title(f"{title} ({len(boxes)} detections)", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Color map for different classes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Draw each detection
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Get class name
        class_name = class_names.get(int(label), f"class_{label}")
        
        # Choose color based on class
        color = colors[int(label) % len(colors)]
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        # Draw label with background
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    plt.tight_layout()
    return fig


def save_comparison_visualization(
    image_path,
    pred_before,
    pred_after,
    class_names,
    output_path,
    score_threshold=0.3
):
    """
    Save side-by-side comparison of before and after ETA.
    
    Args:
        image_path: path to original image
        pred_before: predictions dict before adaptation
        pred_after: predictions dict after adaptation
        class_names: dict mapping class IDs to names
        output_path: where to save the comparison
        score_threshold: minimum confidence to display
    """
    # Load image
    image = cv2.imread(image_path)
    
    # Create side-by-side figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Color map
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # ========== BEFORE ETA ==========
    ax1.imshow(image_rgb)
    keep_before = pred_before['scores'] >= score_threshold
    boxes_before = pred_before['boxes'][keep_before].cpu().numpy()
    labels_before = pred_before['labels'][keep_before].cpu().numpy()
    scores_before = pred_before['scores'][keep_before].cpu().numpy()
    
    for box, label, score in zip(boxes_before, labels_before, scores_before):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        class_name = class_names.get(int(label), f"class_{label}")
        color = colors[int(label) % len(colors)]
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax1.add_patch(rect)
        
        label_text = f"{class_name}"
        ax1.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    ax1.set_title(f"Before ETA ({len(boxes_before)} detections)", fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # ========== AFTER ETA ==========
    ax2.imshow(image_rgb)
    keep_after = pred_after['scores'] >= score_threshold
    boxes_after = pred_after['boxes'][keep_after].cpu().numpy()
    labels_after = pred_after['labels'][keep_after].cpu().numpy()
    scores_after = pred_after['scores'][keep_after].cpu().numpy()
    
    for box, label, score in zip(boxes_after, labels_after, scores_after):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        class_name = class_names.get(int(label), f"class_{label}")
        color = colors[int(label) % len(colors)]
        
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none'
        )
        ax2.add_patch(rect)
        
        label_text = f"{class_name}"
        ax2.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7),
            fontsize=9,
            color='white',
            weight='bold'
        )
    
    ax2.set_title(f"After ETA ({len(boxes_after)} detections)", fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved comparison to: {output_path}")

def bn_energy_adaptation(
    model,
    energy_model,
    dataset,
    device,
    adaptation_lr,
    iterations_per_image=3,
    score_threshold=0.05,
    viz_score_threshold=0.3,  # Higher threshold for visualizations
    save_viz_every=50,  # Save visualization every N images
    output_dir='visualizations',
    writer=None,
    verbose=False
):
    """
    Sequential test-time adaptation with visualization outputs.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Class names
    class_names = {
        0: "person",
        1: "rider", 
        2: "car",
        3: "truck",
        4: "bus",
        5: "train",
        6: "motorcycle",
        7: "bicycle"
    }
    
    # Setup optimizer for BN affine parameters
    bn_params = []
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.weight is not None:
                bn_params.append(module.weight)
            if module.bias is not None:
                bn_params.append(module.bias)
    
    optimizer = optim.SGD(bn_params, lr=adaptation_lr, momentum=0.9)
    print(f"Optimizing {len(bn_params)} BN affine parameters")
    
    # Save initial BN state
    initial_bn_state = save_bn_state(model)
    
    # Accumulate ALL predictions and ground truths for global mAP
    all_preds_before = []
    all_preds_after = []
    all_gts = []
    
    results = {
        'energy_losses': [],
        'num_detections_before': [],
        'num_detections_after': []
    }
    
    global_step = 0
    
    # Process each image sequentially
    for img_idx, data in enumerate(dataset):
        print(f"[Image {img_idx + 1}/{len(dataset)}] {Path(data['image_path']).name}")
        
        # Reset BN to initial state
        restore_bn_state(model, initial_bn_state)
        
        # Load image
        image = cv2.imread(data['image_path'])
        if image is None:
            print(f"  ⚠️  Failed to load image, skipping...")
            continue
        
        # Prepare ground truth
        gt = prepare_gt_instances(
            data['annotations'],
            (data['height'], data['width']),
            device
        )
        
        # Prepare image tensor
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
        
        # ========================================
        # Evaluate BEFORE adaptation
        # ========================================
        model.eval()
        set_bn_to_eval(model)
        
        with torch.no_grad():
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = model(inputs)
            predictions_before = outputs[0]['instances']
        
        pred_before = filter_and_remap_predictions(predictions_before, score_threshold)
        
        # ========================================
        # Perform BN adaptation with energy guidance
        # ========================================
        model.eval()
        set_bn_to_train(model)
        
        total_energy_loss = 0
        for iter_idx in range(iterations_per_image):
            optimizer.zero_grad()
            energy_loss = compute_energy_loss(
                model, energy_model, image_tensor, height, width, device
            )
            
            if torch.isfinite(energy_loss) and energy_loss.item() > 0:
                energy_loss.backward()
                optimizer.step()
                total_energy_loss += energy_loss.item()
                
                if writer is not None:
                    writer.add_scalar('Energy/iter_loss', energy_loss.item(), global_step)
                    global_step += 1
        
        avg_energy_loss = total_energy_loss / iterations_per_image if iterations_per_image > 0 else 0
        
        # ========================================
        # Evaluate AFTER adaptation
        # ========================================
        model.eval()
        set_bn_to_eval(model)
        
        with torch.no_grad():
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = model(inputs)
            predictions_after = outputs[0]['instances']
        
        pred_after = filter_and_remap_predictions(predictions_after, score_threshold)
        
        # ========================================
        # Save visualization
        # ========================================
        if img_idx % save_viz_every == 0 or img_idx < 5:  # Always save first 5
            output_path = output_dir / f"comparison_{img_idx:04d}.png"
            save_comparison_visualization(
                data['image_path'],
                pred_before,
                pred_after,
                class_names,
                str(output_path),
                score_threshold=viz_score_threshold
            )
        
        # Accumulate for global mAP computation
        all_preds_before.append(pred_before)
        all_preds_after.append(pred_after)
        all_gts.append(gt)
        
        # Store per-image metrics
        results['energy_losses'].append(avg_energy_loss)
        results['num_detections_before'].append(len(pred_before['boxes']))
        results['num_detections_after'].append(len(pred_after['boxes']))
        
        # Print per-image info
        print(f"  ⚡ Energy loss: {avg_energy_loss:.4f}")
        print(f"  📦 Detections: {results['num_detections_before'][-1]} → {results['num_detections_after'][-1]}")
        
        # Compute and log intermediate global mAP every 50 images
        if (img_idx + 1) % 50 == 0 or img_idx == 0:
            intermediate_map_before = compute_map(all_preds_before, all_gts, verbose=False)
            intermediate_map_after = compute_map(all_preds_after, all_gts, verbose=False)
            print(f"  📊 Global mAP so far ({img_idx + 1} images): {intermediate_map_before:.3f} → {intermediate_map_after:.3f} (Δ: {intermediate_map_after - intermediate_map_before:+.3f})")
            
            if writer is not None:
                writer.add_scalar('mAP_global/before', intermediate_map_before, img_idx)
                writer.add_scalar('mAP_global/after', intermediate_map_after, img_idx)
                writer.add_scalar('mAP_global/improvement', intermediate_map_after - intermediate_map_before, img_idx)
        
        print()
    
    # Final mAP computation
    print("=" * 60)
    print("Computing Global mAP...")
    print("=" * 60)
    
    map_before_global = compute_map(all_preds_before, all_gts, verbose=True)
    print()
    map_after_global = compute_map(all_preds_after, all_gts, verbose=True)
    
    results['map_before_global'] = map_before_global
    results['map_after_global'] = map_after_global
    
    print()
    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Global mAP before:   {map_before_global:.4f}")
    print(f"Global mAP after:    {map_after_global:.4f}")
    print(f"Global improvement:  {map_after_global - map_before_global:+.4f}")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 60)
    
    return results

# ==============================================================
# MAIN
# ==============================================================

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("BN + Energy Adaptation for Object Detection")
    print("=" * 60)
    print()
    
    # Configuration
    CORRUPTION = "motion_blur"
    config = {
        # MOTION BLURRED IMAGES DIRECTORY
        'image_dir': f'/home/lyz6/AMROD/datasets/{CORRUPTION}/leftImg8bit/val',
        'annotation_file': '/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        
        # Energy model path - trained using new_energy_model.py
        'energy_model_path': f'models/{CORRUPTION}_roi_energy_model_epoch2.pth',
        
        # Optional: Load from previous checkpoint instead of COCO weights
        'checkpoint_path': None,
        'max_images': 5,
        'adaptation_lr': 2,  # Learning rate for BN affine parameters
        'iterations_per_image': 5,
        'score_threshold': 0.05,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    # Learning Rates:
    # 0.2 for SNOW
    # 2 for MOTION
    # 0,3 for FOG
    # for DEFOCUS
    # for BRIGHTNESS
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    print(f"Using CORRUPTED images from: {config['image_dir']}\n")
    
    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f'runs/bn_energy_adaptation_{timestamp}'
    writer = SummaryWriter(log_dir)
    print(f"📊 TensorBoard logs: {log_dir}")
    print(f"   View with: tensorboard --logdir={log_dir}\n")
    
    # Setup Detectron2 with BN adaptation
    print("[1/4] Setting up Detectron2 with BN adaptation...")
    model, cfg = setup_detectron2_for_bn_adaptation(
        device,
        checkpoint_path=config['checkpoint_path']
    )
    print("✅ Setup complete with BN layers ready for adaptation\n")
    
    # Load energy model
    print("[2/4] Loading trained energy model...")
    from new_energy_model import ROI_EnergyModel
    
    # Energy model expects 256-channel ROI features (from FPN)
    print(f"   Energy model: ROI-based, expects 256-channel 7x7 features")
    
    energy_model = ROI_EnergyModel(in_channels=256).to(device)
    
    if os.path.exists(config['energy_model_path']):
        checkpoint = torch.load(config['energy_model_path'], map_location=device)
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            energy_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            energy_model.load_state_dict(checkpoint)
        print(f"✅ Loaded energy model from {config['energy_model_path']}\n")
    else:
        print(f"⚠️  Energy model not found at {config['energy_model_path']}")
        print("   Using randomly initialized energy model (for testing only!)\n")
    
    energy_model.eval()
    for param in energy_model.parameters():
        param.requires_grad = False
    
    # Load dataset
    print("[3/4] Loading dataset...")
    dataset = load_cityscapes_data(
        config['image_dir'],
        config['annotation_file'],
        max_images=config['max_images']
    )
    
    if len(dataset) == 0:
        print("\n❌ ERROR: No images loaded! Cannot proceed.")
        return
    
    print()
    
    # Run BN + Energy adaptation
    print("[4/4] Running BN adaptation with energy guidance...")
    print(f"Adaptation LR: {config['adaptation_lr']}")
    print(f"Iterations per image: {config['iterations_per_image']}")
    print(f"Resetting BN to initial state before each image")
    print()
    
    results = bn_energy_adaptation(
        model=model,
        energy_model=energy_model,
        dataset=dataset,
        device=device,
        adaptation_lr=config['adaptation_lr'],
        iterations_per_image=config['iterations_per_image'],
        score_threshold=config['score_threshold'],
        writer=writer,
        verbose=True
    )
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"📊 TensorBoard logs saved to: {log_dir}")


if __name__ == '__main__':
    main()