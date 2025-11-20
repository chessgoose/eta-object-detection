"""
Detectron2 Sequential Test-Time Adaptation with Energy Model
"""

"""
As high-energy test-time predictions
ˆ dt = fθ,ψ(It, zt) correspond to out-of-distribution
predictions, we posit that minimizing their energy will improve
the fidelity of predictions by aligning them back to
those of the source data distribution. Hence, minimizing the
energy e predicted from Eϕ is analogous to minimizing the
likelihood of error learned from the source data. This motivates
an energy-based adaptation loss that aims to reduce
the energy of ˆ dt(x) conditioned on zt(x):
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

from detectron2.modeling import build_model, BACKBONE_REGISTRY, Backbone
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone import build_backbone

import warnings
warnings.filterwarnings('ignore')


# ==============================================================
# CLASS MAPPING
# ==============================================================

CITYSCAPES_TO_COCO = {
    1: 0,  # person → person
    2: 0,  # rider → person
    3: 2,  # car → car
    4: 1,  # bicycle → bicycle
    5: 3,  # motorcycle → motorcycle
    6: 5,  # bus → bus
    7: 7,  # truck → truck
    8: 6,  # train → train
}

COCO_CITYSCAPES_CLASSES = {
    0: 0,   # person
    1: 1,   # bicycle
    2: 2,   # car
    3: 3,   # motorcycle
    5: 5,   # bus
    6: 6,   # train
    7: 7,   # truck
}

EVAL_CLASSES = [0, 1, 2, 3, 5, 6, 7]


# ==============================================================
# LIGHTWEIGHT ADAPTATION LAYER
# ==============================================================

class LightweightAdaptationLayer(nn.Module):
    """Lightweight adaptation module for test-time adaptation"""
    def __init__(self, in_channels: int):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Initialize adaptation layer near identity
        for m in self.adapt.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.zeros_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return x + self.adapt(x)  # Residual connection


# ==============================================================
# CUSTOM BACKBONE WITH ADAPTATION LAYER
# ==============================================================

@BACKBONE_REGISTRY.register()
class AdaptiveBackbone(Backbone):
    """
    Wrapper backbone that adds an adaptation layer to a specific feature level.
    This backbone wraps an existing backbone and applies adaptation to one feature map.
    """
    def __init__(self, cfg, input_shape, base_backbone, feature_key='p4'):
        super().__init__()
        
        self.base_backbone = base_backbone
        self.feature_key = feature_key
        
        # Get the shape of the feature we want to adapt
        base_output_shape = base_backbone.output_shape()
        
        if feature_key not in base_output_shape:
            # If requested feature doesn't exist, use first available
            available_keys = list(base_output_shape.keys())
            print(f"⚠️  Feature '{feature_key}' not found. Available: {available_keys}")
            self.feature_key = available_keys[0]
            print(f"   Using '{self.feature_key}' instead")
        
        feature_channels = base_output_shape[self.feature_key].channels
        
        # Create adaptation layer
        self.adaptation_layer = LightweightAdaptationLayer(feature_channels)
        print(f"✅ Created adaptation layer for '{self.feature_key}' ({feature_channels} channels)")
        
        # Store output shape (same as base backbone)
        self._output_shape = base_output_shape
    
    def forward(self, x):
        """Forward pass with adaptation applied to specified feature"""
        # Get features from base backbone
        features = self.base_backbone(x)
        
        # Apply adaptation to the specified feature
        if self.feature_key in features:
            features[self.feature_key] = self.adaptation_layer(features[self.feature_key])
        
        return features
    
    def output_shape(self):
        """Return output shape (same as base backbone)"""
        return self._output_shape


# ==============================================================
# DETECTRON2 SETUP WITH ADAPTIVE BACKBONE
# ==============================================================

def setup_detectron2_with_adaptation(device="cuda", feature_key='p4', 
                                     checkpoint_path=None):
    """Setup Detectron2 model with adaptation layer inserted in backbone.
    
    Args:
        device: Device to load model on
        feature_key: Which feature map to adapt (e.g., 'p4')
        checkpoint_path: Optional path to load weights from a previous checkpoint.
                        If None, loads COCO pretrained weights.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    # First, build the base backbone
    base_backbone = build_backbone(cfg)
    base_backbone = base_backbone.to(device)
    
    # Test to see what features are available
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 800, 800).to(device)
        dummy_features = base_backbone(dummy_input)
    
    print(f"📊 Available backbone features: {list(dummy_features.keys())}")
    
    # Choose feature to adapt (try p4 first, then others)
    available_keys = list(dummy_features.keys())
    if feature_key not in available_keys:
        for key in ['p4', 'p3', 'p5', 'res4', 'res5']:
            if key in available_keys:
                feature_key = key
                break
        else:
            feature_key = available_keys[0]
    
    # Create adaptive backbone wrapper
    adaptive_backbone = AdaptiveBackbone(
        cfg=cfg,
        input_shape=None,
        base_backbone=base_backbone,
        feature_key=feature_key
    ).to(device)
    
    # Build the full model with our custom backbone
    # We need to manually construct the model
    from detectron2.modeling.meta_arch import GeneralizedRCNN
    from detectron2.modeling.proposal_generator import build_proposal_generator
    from detectron2.modeling.roi_heads import build_roi_heads
    
    model = GeneralizedRCNN(
        backbone=adaptive_backbone,
        proposal_generator=build_proposal_generator(cfg, adaptive_backbone.output_shape()),
        roi_heads=build_roi_heads(cfg, adaptive_backbone.output_shape()),
        pixel_mean=cfg.MODEL.PIXEL_MEAN,
        pixel_std=cfg.MODEL.PIXEL_STD,
    )
    
    model = model.to(device)
    
    # Load pretrained weights using DetectionCheckpointer (same as original code)
    checkpointer = DetectionCheckpointer(model)
    
    # Determine which weights to load
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading weights from custom checkpoint: {checkpoint_path}")
        weights_path = checkpoint_path
    else:
        print("   Loading COCO pretrained weights...")
        weights_path = cfg.MODEL.WEIGHTS
    
    # Load with checkpointer - it handles remapping automatically
    # Use strict=False to allow missing keys (adaptation layer)
    loaded_dict = checkpointer._load_file(weights_path)
    
    # Remap backbone keys: backbone.* -> backbone.base_backbone.*
    if 'model' in loaded_dict:
        state_dict = loaded_dict['model']
    else:
        state_dict = loaded_dict
    
    remapped_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.') and not k.startswith('backbone.base_backbone.'):
            # Remap backbone keys
            new_key = 'backbone.base_backbone.' + k[9:]
            remapped_state_dict[new_key] = v
        else:
            remapped_state_dict[k] = v
    
    # Load the remapped state dict
    incompatible = checkpointer._load_model({"model": remapped_state_dict})
    
    # Print loading info
    if incompatible is not None:
        adaptation_missing = [k for k in incompatible.missing_keys if 'adaptation_layer' in k]
        other_missing = [k for k in incompatible.missing_keys if 'adaptation_layer' not in k]
        
        print(f"   ✅ Loaded weights successfully")
        print(f"   - Adaptation layer keys (expected missing): {len(adaptation_missing)}")
        if other_missing:
            print(f"   ⚠️  Other missing keys: {len(other_missing)}")
        if incompatible.unexpected_keys:
            print(f"   ⚠️  Unexpected keys: {len(incompatible.unexpected_keys)}")
    
    # Freeze all parameters except adaptation layer
    for param in model.parameters():
        param.requires_grad = False
    
    for param in adaptive_backbone.adaptation_layer.parameters():
        param.requires_grad = True
    
    # Set appropriate modes
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    # Only adaptation layer's BN in training mode
    for module in adaptive_backbone.adaptation_layer.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.train()

    model.eval()
    
    # Store reference to adaptation layer for easy access
    model.adaptation_layer = adaptive_backbone.adaptation_layer
    model.adaptation_feature_key = feature_key
    
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
        if ann['category_id'] in CITYSCAPES_TO_COCO:
            ann_copy = ann.copy()
            # Map to Cityscapes→COCO class
            ann_copy['category_id'] = CITYSCAPES_TO_COCO[ann['category_id']]
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
    """Filter predictions to only keep Cityscapes-relevant COCO classes."""
    if not isinstance(predictions, Instances):
        return predictions
    
    pred_boxes = predictions.pred_boxes.tensor
    pred_classes = predictions.pred_classes
    pred_scores = predictions.scores
    
    # Filter by score
    score_mask = pred_scores > score_threshold
    
    # Filter by relevant classes
    class_mask = torch.zeros_like(score_mask)
    for coco_class in COCO_CITYSCAPES_CLASSES.keys():
        class_mask |= (pred_classes == coco_class)
    
    # Combined mask
    keep_mask = score_mask & class_mask
    
    # Apply mask
    filtered_boxes = pred_boxes[keep_mask]
    filtered_classes = pred_classes[keep_mask]
    filtered_scores = pred_scores[keep_mask]
    
    # Remap classes to evaluation IDs
    remapped_classes = filtered_classes.clone()
    for coco_id, eval_id in COCO_CITYSCAPES_CLASSES.items():
        remapped_classes[filtered_classes == coco_id] = eval_id
    
    return {
        'boxes': filtered_boxes,
        'labels': remapped_classes,
        'scores': filtered_scores
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


# ==============================================================
# mAP COMPUTATION HELPERS
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
    for class_id in EVAL_CLASSES:
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
# ADAPTATION STEP
# ==============================================================

def adaptation_step(model, energy_model, image, device):
    """
    Perform one adaptation step using energy loss.
    The adaptation layer is now inside the backbone, so we just do forward pass normally.
    """
    height, width = image.shape[:2]
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
    
    # Forward pass (adaptation is applied inside backbone)
    inputs = [{"image": image_tensor, "height": height, "width": width}]
    images = model.preprocess_image(inputs)
    features = model.backbone(images.tensor)
    
    # Extract the adapted feature for energy computation
    feature_key = model.adaptation_feature_key
    adapted_features = features[feature_key]
    
    # Compute energy
    energy = energy_model(adapted_features)
    energy_loss = energy.mean()
    
    return energy_loss, features


# ==============================================================
# SEQUENTIAL TEST-TIME ADAPTATION
# ==============================================================

def sequential_test_time_adaptation(
    model,
    energy_model,
    dataset,
    device,
    adaptation_lr=0.001,
    iterations_per_image=3,
    score_threshold=0.05,
    verbose=False
):
    """Sequential test-time adaptation with adaptation layer inside backbone."""
    if not hasattr(model, 'adaptation_layer'):
        raise ValueError("Model must have adaptation_layer attribute")
    
    optimizer = optim.Adam(model.adaptation_layer.parameters(), lr=adaptation_lr)
    
    results = {
        'map_before': [],
        'map_after': [],
        'energy_losses': [],
        'num_detections_before': [],
        'num_detections_after': []
    }
    
    # To evaluate "before" adaptation, we need to temporarily disable the adaptation layer
    # We'll do this by saving/restoring the adaptation layer's parameters
    
    # Process each image sequentially
    for img_idx, data in enumerate(dataset):
        print(f"[Image {img_idx + 1}/{len(dataset)}] {Path(data['image_path']).name}")
        
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
        
        # Evaluate BEFORE adaptation - temporarily make adaptation layer identity
        # Save current adaptation weights
        saved_state = {name: param.clone() for name, param in model.adaptation_layer.named_parameters()}
        
        # Zero out the adaptation layer to make it identity
        with torch.no_grad():
            for module in model.adaptation_layer.adapt.modules():
                if isinstance(module, nn.Conv2d):
                    module.weight.zero_()
                    if module.bias is not None:
                        module.bias.zero_()
        
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
        
        with torch.no_grad():
            height, width = image.shape[:2]
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
            
            # Forward pass (adaptation layer is now identity)
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            
            proposals, _ = model.proposal_generator(images, features, None)
            predictions_before, _ = model.roi_heads(images, features, proposals, None)
            predictions_before = predictions_before[0]
        
        # Restore adaptation layer weights
        with torch.no_grad():
            for name, param in model.adaptation_layer.named_parameters():
                param.copy_(saved_state[name])
        
        # Filter and remap predictions
        pred_before = filter_and_remap_predictions(predictions_before, score_threshold)
        
        # Compute mAP before
        map_before = compute_map([pred_before], [gt], verbose=(verbose and img_idx == 0))
        
        # Debug info
        if verbose and img_idx == 0:
            print(f"\n    DEBUG - Ground truth classes: {gt['labels'].cpu().tolist()}")
            if len(pred_before['labels']) > 0:
                print(f"    DEBUG - Prediction classes: {pred_before['labels'].cpu().tolist()}")
                print(f"    DEBUG - Prediction scores: {pred_before['scores'].cpu().tolist()[:5]}")
            else:
                print(f"    DEBUG - No predictions after filtering!")
            print(f"    DEBUG - Num GTs: {len(gt['boxes'])}, Num Preds: {len(pred_before['boxes'])}\n")
        
        # Perform adaptation iterations
        # Set adaptation layer BN to training mode
        for module in model.adaptation_layer.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()
        
        total_energy_loss = 0
        for iter_idx in range(iterations_per_image):
            optimizer.zero_grad()
            
            # Adaptation step (backbone will apply adaptation layer)
            energy_loss, _ = adaptation_step(model, energy_model, image, device)
            
            if verbose and iter_idx == 0:
                print(f"    DEBUG - Energy loss: {energy_loss.item():.4f}, requires_grad={energy_loss.requires_grad}")
            
            # Backward and optimize
            if torch.isfinite(energy_loss) and energy_loss.item() > 0:
                energy_loss.backward()
                optimizer.step()
                total_energy_loss += energy_loss.item()
            else:
                if verbose:
                    print(f"    Skipping iteration {iter_idx + 1} due to invalid loss")
        
        avg_energy_loss = total_energy_loss / iterations_per_image if iterations_per_image > 0 else 0
        
        # Evaluate AFTER adaptation (with trained adaptation layer)
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()
        
        with torch.no_grad():
            # Forward pass (adaptation layer active)
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            
            proposals, _ = model.proposal_generator(images, features, None)
            predictions_after, _ = model.roi_heads(images, features, proposals, None)
            predictions_after = predictions_after[0]
        
        # Filter and remap predictions
        pred_after = filter_and_remap_predictions(predictions_after, score_threshold)
        
        # Compute mAP after
        map_after = compute_map([pred_after], [gt], verbose=False)
        
        # Store results
        results['map_before'].append(map_before)
        results['map_after'].append(map_after)
        results['energy_losses'].append(avg_energy_loss)
        results['num_detections_before'].append(len(pred_before['boxes']))
        results['num_detections_after'].append(len(pred_after['boxes']))
        
        # Print results
        map_delta = map_after - map_before
        print(f"  📊 mAP: {map_before:.3f} → {map_after:.3f} (Δ: {map_delta:+.3f})")
        print(f"  ⚡ Energy loss: {avg_energy_loss:.4f}")
        print(f"  📦 Detections: {results['num_detections_before'][-1]} → {results['num_detections_after'][-1]}")
        print()
    
    # Print final summary
    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Average mAP before:  {np.mean(results['map_before']):.4f}")
    print(f"Average mAP after:   {np.mean(results['map_after']):.4f}")
    print(f"Average improvement: {np.mean(results['map_after']) - np.mean(results['map_before']):+.4f}")
    print(f"Average energy loss: {np.mean(results['energy_losses']):.4f}")
    print("=" * 60)
    
    return results


# ==============================================================
# MAIN
# ==============================================================

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("ETA for Object Detection - Motion Blurred Images")
    print("WITH ADAPTATION LAYER INSIDE RESNET BACKBONE")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        # MOTION BLURRED IMAGES DIRECTORY
        'image_dir': '/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val',
        'annotation_file': '/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        'energy_model_path': '/home/lyz6/palmer_scratch/eta-object-detection/multiple_roi_energy_model_epoch2.pth',
        
        # Optional: Load from previous checkpoint instead of COCO weights
        # Set to None to use COCO pretrained weights
        'checkpoint_path': None,  # e.g., '/path/to/your/previous_model.pth'
        
        'max_images': 20,
        'adaptation_lr': 1e-2,
        'iterations_per_image': 3,
        'score_threshold': 0.05,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feature_key': 'p4'  # Which feature to adapt
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    print(f"Using MOTION BLURRED images from: {config['image_dir']}\n")
    
    # Setup Detectron2 with adaptive backbone
    print("[1/4] Setting up Detectron2 with adaptive backbone...")
    model, cfg = setup_detectron2_with_adaptation(
        device, 
        feature_key=config['feature_key'],
        checkpoint_path=config['checkpoint_path']
    )
    print("✅ Setup complete with adaptation layer inside backbone\n")
    
    # Load energy model
    print("[2/4] Loading trained energy model...")
    from new_energy_model import ROI_EnergyModel
    
    # Determine energy model channels based on adapted feature
    energy_channels = model.adaptation_layer.adapt[0].in_channels
    print(f"   Energy model input channels: {energy_channels}")
    
    energy_model = ROI_EnergyModel(in_channels=energy_channels).to(device)
    
    if os.path.exists(config['energy_model_path']):
        checkpoint = torch.load(config['energy_model_path'], map_location=device)
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
    
    # Run sequential adaptation
    print("[4/4] Running sequential test-time adaptation...")
    
    # Count adaptable parameters
    adaptable_params = [p for p in model.adaptation_layer.parameters() if p.requires_grad]
    print(f"Adaptable parameters: {len(adaptable_params)}")
    print(f"Total adaptable param count: {sum(p.numel() for p in adaptable_params):,}")
    print()
    
    results = sequential_test_time_adaptation(
        model=model,
        energy_model=energy_model,
        dataset=dataset,
        device=device,
        adaptation_lr=config['adaptation_lr'],
        iterations_per_image=config['iterations_per_image'],
        score_threshold=config['score_threshold'],
        verbose=True
    )
    
    # Save results
    output_file = 'sequential_adaptation_results_backbone_motion_blur.json'
    with open(output_file, 'w') as f:
        json_results = {k: [float(v) for v in vals] for k, vals in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    main()