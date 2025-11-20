"""
Detectron2 Sequential Test-Time Adaptation with Energy Model
ADAPTATION LAYER: inserted AFTER res4 (before FPN fusion)
WITH DIAGNOSTIC TOOLS
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
from detectron2.layers import ShapeSpec

# Additional imports to build resnet bottom-up and FPN
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.fpn import FPN

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
# LIGHTWEIGHT ADAPTATION LAYER (1x1 CONV FOR TRUE IDENTITY)
# ==============================================================

class LightweightAdaptationLayer(nn.Module):
    """Lightweight adaptation module initialized as perfect identity.
    
    Uses 1x1 convolution for guaranteed identity transformation.
    """
    def __init__(self, in_channels: int):
        super().__init__()
        # Use 1x1 conv for guaranteed identity (no spatial mixing)
        self.adapt = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        
        # Perfect identity initialization: diagonal matrix
        with torch.no_grad():
            self.adapt.weight.zero_()
            for i in range(in_channels):
                self.adapt.weight[i, i, 0, 0] = 1.0

    def forward(self, x):
        return self.adapt(x)


# ==============================================================
# BOTTOM-UP WRAPPER: inject adaptation AFTER res4, BEFORE FPN
# ==============================================================

class BottomUpWithRes4Adapt(Backbone):
    """
    Wrap a ResNet bottom-up (res2..res5) and apply adaptation to 'res4' output.
    This wrapper preserves module names (res2, res3, res4, res5) so checkpoint keys
    like 'backbone.bottom_up.res4.X' map correctly.
    """
    def __init__(self, resnet_backbone, adapt_channels=1024):
        """
        resnet_backbone: a ResNet bottom-up module (returned by build_resnet_backbone)
        adapt_channels: number of channels for res4 (usually 1024 for ResNet50)
        """
        super().__init__()
        # keep internal pointer to original resnet
        self.resnet = resnet_backbone

        # expose stage modules under the wrapper with identical names so state dict keys still match
        # These are references (no copy) so parameters are shared
        if hasattr(self.resnet, "res2"):
            self.res2 = self.resnet.res2
        if hasattr(self.resnet, "res3"):
            self.res3 = self.resnet.res3
        if hasattr(self.resnet, "res4"):
            self.res4 = self.resnet.res4
        if hasattr(self.resnet, "res5"):
            self.res5 = self.resnet.res5

        # Copy over important attributes from the original resnet backbone
        # These are needed by Detectron2's Backbone interface
        if hasattr(self.resnet, "size_divisibility"):
            self._size_divisibility = self.resnet.size_divisibility
        else:
            self._size_divisibility = 32
            
        if hasattr(self.resnet, "_out_features"):
            self._out_features = self.resnet._out_features
        else:
            self._out_features = ["res2", "res3", "res4", "res5"]
            
        if hasattr(self.resnet, "_out_feature_strides"):
            self._out_feature_strides = self.resnet._out_feature_strides
        else:
            self._out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
            
        if hasattr(self.resnet, "_out_feature_channels"):
            self._out_feature_channels = self.resnet._out_feature_channels
        else:
            self._out_feature_channels = {"res2": 256, "res3": 512, "res4": 1024, "res5": 2048}

        # Insert adaptation layer that will operate on the output of res4
        self.adaptation_layer = LightweightAdaptationLayer(adapt_channels)
        # name for easy access in higher-level code
        self.adaptation_stage_name = "res4"

    def forward(self, x):
        """
        Run the original resnet bottom-up and apply adaptation to res4 feature.
        Returns a dict: {'res2':..., 'res3':..., 'res4':..., 'res5':...}
        """
        # The original resnet object (self.resnet) implements __call__/forward to return a dict
        res_features = self.resnet(x)

        # Apply adaptation on res4 (if present)
        if "res4" in res_features:
            res_features["res4"] = self.adaptation_layer(res_features["res4"])
        else:
            # fallback: try to detect keys like 'res3'/'res5' - but main path is res4
            keys = list(res_features.keys())
            if len(keys) >= 3:
                # choose the middle key if res4 is missing
                k = keys[len(keys) // 2]
                res_features[k] = self.adaptation_layer(res_features[k])

        return res_features

    def output_shape(self):
        # Provide shape mapping consistent with ResNet bottom-up
        return self.resnet.output_shape()


# ==============================================================
# DETECTRON2 SETUP WITH ADAPTATION LAYER AFTER res4
# ==============================================================

def setup_detectron2_with_adaptation(device="cuda", checkpoint_path=None, feature_key='p4'):
    """Setup Detectron2 model with adaptation layer inserted after res4 (before FPN)."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    # Build a ResNet bottom-up backbone (without FPN)
    print("   Building ResNet bottom-up (res2..res5)...")
    input_shape = ShapeSpec(channels=3)
    resnet_bottom_up = build_resnet_backbone(cfg, input_shape=input_shape)

    # Determine res4 channels from output shape
    resnet_out_shape = resnet_bottom_up.output_shape()
    if 'res4' in resnet_out_shape:
        res4_channels = resnet_out_shape['res4'].channels
    else:
        res4_channels = 1024
    print(f"   res4 channels detected: {res4_channels}")

    # Wrap bottom-up with adaptation (applies adaptation to res4 feature)
    bottom_up_wrapped = BottomUpWithRes4Adapt(resnet_bottom_up, adapt_channels=res4_channels)

    # Build FPN on top of our wrapped bottom-up
    print("   Building FPN on top of adapted bottom-up...")
    from detectron2.modeling.backbone.fpn import LastLevelMaxPool
    
    fpn = FPN(
        bottom_up=bottom_up_wrapped,
        in_features=["res2", "res3", "res4", "res5"],
        out_channels=256,
        norm="",
        top_block=LastLevelMaxPool(),
        fuse_type="sum"
    )

    # Build the full model using GeneralizedRCNN with this custom backbone
    from detectron2.modeling.meta_arch import GeneralizedRCNN
    from detectron2.modeling.proposal_generator import build_proposal_generator
    from detectron2.modeling.roi_heads import build_roi_heads

    model = GeneralizedRCNN(
        backbone=fpn,
        proposal_generator=build_proposal_generator(cfg, fpn.output_shape()),
        roi_heads=build_roi_heads(cfg, fpn.output_shape()),
        pixel_mean=cfg.MODEL.PIXEL_MEAN,
        pixel_std=cfg.MODEL.PIXEL_STD,
    )

    model = model.to(device)

    # Load pretrained weights using DetectionCheckpointer
    checkpointer = DetectionCheckpointer(model)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"   Loading weights from custom checkpoint: {checkpoint_path}")
        checkpointer.load(checkpoint_path)
    else:
        print("   Loading COCO pretrained weights...")
        checkpointer.load(cfg.MODEL.WEIGHTS)

    print(f"   ✅ Loaded weights using DetectionCheckpointer")

    # Freeze all params except adaptation layer
    for param in model.parameters():
        param.requires_grad = False

    # adaptation layer lives at: model.backbone.bottom_up.adaptation_layer
    try:
        adapt_layer = model.backbone.bottom_up.adaptation_layer
        for param in adapt_layer.parameters():
            param.requires_grad = True
    except Exception as e:
        print("   ⚠️  Could not locate adaptation_layer at model.backbone.bottom_up.adaptation_layer")
        print("   Exception:", e)
        adapt_layer = None

    # Set appropriate modes: everything eval except adaptation BN layers
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()

    if adapt_layer is not None:
        for module in adapt_layer.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.train()

    # Store references for downstream code
    model.adaptation_layer = adapt_layer
    model.adaptation_feature_key = feature_key
    model.adaptation_stage = 'res4'

    return model, cfg


# ==============================================================
# DIAGNOSTIC: TEST IDENTITY INITIALIZATION
# ==============================================================

def test_identity_initialization(model, device):
    """Test that adaptation layer is truly identity."""
    print("\n" + "=" * 60)
    print("TESTING IDENTITY INITIALIZATION")
    print("=" * 60)
    
    with torch.no_grad():
        # Create test input matching res4 output size
        test_input = torch.randn(1, 1024, 32, 32, device=device)
        
        # Get output through adaptation layer
        test_output = model.backbone.bottom_up.adaptation_layer(test_input)
        
        # Check if output ≈ input
        diff = (test_output - test_input).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"Max difference from identity: {max_diff:.10f}")
        print(f"Mean difference from identity: {mean_diff:.10f}")
        
        if max_diff > 1e-5:
            print("❌ WARNING: Adaptation layer is NOT acting as identity!")
            print(f"   Expected ~0, got {max_diff}")
            return False
        else:
            print("✅ Adaptation layer is properly initialized as identity")
            return True


# ==============================================================
# DIAGNOSTIC: TEST PRETRAINED BASELINE
# ==============================================================

def test_pretrained_baseline(model, dataset, device):
    """Test model performance without any adaptation layer interference."""
    print("\n" + "=" * 60)
    print("TESTING PRETRAINED BASELINE (BYPASSING ADAPTATION)")
    print("=" * 60)
    
    # Test on first image
    data = dataset[0]
    image = cv2.imread(data['image_path'])
    if image is None:
        print("Failed to load test image")
        return
    
    height, width = image.shape[:2]
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
    
    # Test 1: With identity adaptation layer
    print("\n[Test 1] With identity adaptation layer:")
    with torch.no_grad():
        inputs = [{"image": image_tensor, "height": height, "width": width}]
        outputs = model(inputs)
        predictions = outputs[0]['instances']
    
    print(f"  Total predictions: {len(predictions)}")
    if len(predictions) > 0:
        print(f"  Classes (first 10): {predictions.pred_classes[:10].cpu().tolist()}")
        print(f"  Scores (first 10): {predictions.scores[:10].cpu().tolist()}")
        print(f"  Unique classes: {predictions.pred_classes.unique().cpu().tolist()}")
    
    # Filter predictions
    from copy import deepcopy
    pred_filtered = filter_and_remap_predictions(deepcopy(predictions), score_threshold=0.05)
    print(f"  After filtering: {len(pred_filtered['boxes'])} predictions")
    if len(pred_filtered['labels']) > 0:
        print(f"  Filtered classes: {pred_filtered['labels'].unique().cpu().tolist()}")
    
    print("\n" + "=" * 60)


# ==============================================================
# ROI FEATURE EXTRACTION
# ==============================================================

def extract_roi_features(model, image_tensor, boxes):
    """
    Extract 7x7 ROI features from Detectron2 backbone using multi-level FPN.
    """
    from detectron2.modeling.poolers import ROIPooler
    from detectron2.structures import Boxes

    # Extract FPN features
    features = model.backbone(image_tensor)

    # Multi-level pooler
    pooler = ROIPooler(
        output_size=7,
        scales=tuple(1.0 / model.backbone.output_shape()[k].stride for k in ["p2", "p3", "p4", "p5"]),
        sampling_ratio=0,
        pooler_type="ROIAlignV2"
    )

    # Make list-of-boxes as required by ROIPooler
    roi_boxes = [Boxes(boxes)]
    roi_feats = pooler([features["p2"], features["p3"], features["p4"], features["p5"]], roi_boxes)

    return roi_feats


# ==============================================================
# DATASET LOADING & PREP
# ==============================================================

def load_cityscapes_data(image_dir, annotation_file, max_images=None):
    """Load Cityscapes validation data (COCO-formatted annotation file)."""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Build image_id to annotations mapping
    img_id_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        if ann['category_id'] in CITYSCAPES_TO_COCO:
            ann_copy = ann.copy()
            ann_copy['category_id'] = CITYSCAPES_TO_COCO[ann['category_id']]
            img_id_to_anns[ann['image_id']].append(ann_copy)

    # Build dataset list with resolved image paths
    dataset = []
    for img_info in coco_data['images']:
        file_name = img_info['file_name']
        if '/' in file_name:
            file_name = file_name.split('/')[-1]

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
        labels.append(ann['category_id'])

    return {
        'boxes': torch.tensor(boxes, dtype=torch.float32, device=device),
        'labels': torch.tensor(labels, dtype=torch.int64, device=device)
    }


# ==============================================================
# PREDICTION FILTERING (WITH DEBUG)
# ==============================================================

def filter_and_remap_predictions(predictions, score_threshold=0.05, verbose=False):
    """Filter predictions to only keep Cityscapes-relevant COCO classes."""
    if not isinstance(predictions, Instances):
        return predictions

    pred_boxes = predictions.pred_boxes.tensor
    pred_classes = predictions.pred_classes
    pred_scores = predictions.scores

    if verbose:
        print(f"[FILTER DEBUG] Raw pred classes: {pred_classes.unique().cpu().tolist()}")
        print(f"[FILTER DEBUG] Expected COCO classes: {list(COCO_CITYSCAPES_CLASSES.keys())}")
        print(f"[FILTER DEBUG] Total raw predictions: {len(pred_classes)}")

    # Filter by score
    score_mask = pred_scores > score_threshold

    # Filter by relevant classes
    class_mask = torch.zeros_like(score_mask)
    for coco_class in COCO_CITYSCAPES_CLASSES.keys():
        class_mask |= (pred_classes == coco_class)

    # Combined mask
    keep_mask = score_mask & class_mask

    if verbose:
        print(f"[FILTER DEBUG] Score mask: {score_mask.sum()} / {len(score_mask)}")
        print(f"[FILTER DEBUG] Class mask: {class_mask.sum()} / {len(class_mask)}")
        print(f"[FILTER DEBUG] Keep mask: {keep_mask.sum()} / {len(keep_mask)}")

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
# mAP / IOU HELPERS
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
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

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
    Perform one adaptation step using energy loss on ROI features.
    """
    height, width = image.shape[:2]
    image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)

    # Forward pass to get detections
    inputs = [{"image": image_tensor, "height": height, "width": width}]
    with torch.enable_grad():
        outputs = model(inputs)
    
    predictions = outputs[0]['instances']
    
    # If no predictions, return zero loss
    if len(predictions) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), None
    
    # Get predicted boxes
    pred_boxes = predictions.pred_boxes.tensor
    
    # Extract ROI features
    images = model.preprocess_image(inputs)
    roi_features = extract_roi_features(model, images.tensor, pred_boxes)
    
    # Compute energy on ROI features
    if callable(energy_model):
        energy = energy_model(roi_features)
    else:
        # Fallback
        energy = roi_features.pow(2).mean(dim=(1, 2, 3))
    
    # Ensure energy has valid shape and gradients
    if energy.numel() == 0:
        energy_loss = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        energy_loss = energy.mean() + 1e-8

    return energy_loss, roi_features


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
    """Sequential test-time adaptation with adaptation layer after res4."""
    if not hasattr(model, 'adaptation_layer') or model.adaptation_layer is None:
        raise ValueError("Model must have adaptation_layer attribute (model.adaptation_layer)")

    optimizer = optim.Adam(model.adaptation_layer.parameters(), lr=adaptation_lr)

    results = {
        'map_before': [],
        'map_after': [],
        'energy_losses': [],
        'num_detections_before': [],
        'num_detections_after': []
    }

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

        # Evaluate BEFORE adaptation (with identity - no need to reset, already identity)
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        with torch.no_grad():
            height, width = image.shape[:2]
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = model(inputs)
            predictions_before = outputs[0]['instances']

        # Filter and remap predictions
        pred_before = filter_and_remap_predictions(
            predictions_before, 
            score_threshold,
            verbose=(verbose and img_idx == 0)
        )

        # Compute mAP before
        map_before = compute_map([pred_before], [gt], verbose=(verbose and img_idx == 0))

        # Debug info
        if verbose and img_idx == 0:
            print(f"\n    DEBUG - Ground truth classes: {gt['labels'].cpu().tolist()}")
            if len(pred_before['labels']) > 0:
                print(f"    DEBUG - Prediction classes (before): {pred_before['labels'].cpu().tolist()}")
                print(f"    DEBUG - Prediction scores (before): {pred_before['scores'].cpu().tolist()[:5]}")
            else:
                print(f"    DEBUG - No predictions after filtering!")
            print(f"    DEBUG - Num GTs: {len(gt['boxes'])}, Num Preds: {len(pred_before['boxes'])}\n")

        # Perform adaptation iterations
        total_energy_loss = 0
        successful_iters = 0
        for iter_idx in range(iterations_per_image):
            optimizer.zero_grad()

            # Adaptation step
            energy_loss, _ = adaptation_step(model, energy_model, image, device)

            if verbose and iter_idx == 0:
                print(f"    DEBUG - Energy loss: {energy_loss.item():.4f}, requires_grad={energy_loss.requires_grad}")

            # Backward and optimize with safeguards
            if torch.isfinite(energy_loss) and energy_loss.item() > 1e-6:
                energy_loss.backward()
                
                # Clip gradients to prevent instability
                torch.nn.utils.clip_grad_norm_(model.adaptation_layer.parameters(), max_norm=0.1)
                
                optimizer.step()
                total_energy_loss += energy_loss.item()
                successful_iters += 1
            else:
                if verbose:
                    print(f"    Skipping iteration {iter_idx + 1} due to invalid loss")

        avg_energy_loss = total_energy_loss / successful_iters if successful_iters > 0 else 0

        # Evaluate AFTER adaptation
        model.eval()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                module.eval()

        with torch.no_grad():
            height, width = image.shape[:2]
            image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = model(inputs)
            predictions_after = outputs[0]['instances']

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
    print(f"Average mAP before:  {np.mean(results['map_before']):.4f}" if results['map_before'] else "No before results")
    print(f"Average mAP after:   {np.mean(results['map_after']):.4f}" if results['map_after'] else "No after results")
    if results['map_before'] and results['map_after']:
        print(f"Average improvement: {np.mean(results['map_after']) - np.mean(results['map_before']):+.4f}")
    print(f"Average energy loss: {np.mean(results['energy_losses']):.4f}" if results['energy_losses'] else "No energy losses")
    print("=" * 60)

    return results


# ==============================================================
# MAIN
# ==============================================================

def main():
    """Main function with diagnostics"""
    print("\n" + "=" * 60)
    print("ETA for Object Detection - Motion Blurred Images")
    print("ADAPTATION LAYER: AFTER res4 (before FPN)")
    print("WITH DIAGNOSTIC TOOLS")
    print("=" * 60)
    print()

    # Configuration
    #/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/val'
    config = {
        'image_dir': '/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val',
        'annotation_file': '/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        'energy_model_path': '/home/lyz6/palmer_scratch/eta-object-detection/multiple_roi_energy_model_epoch2.pth',
        'checkpoint_path': None,
        'max_images': 20,
        'adaptation_lr': 5e-5,
        'iterations_per_image': 3,
        'score_threshold': 0.05,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'feature_key': 'p4'
    }

    device = torch.device(config['device'])
    print(f"Using device: {device}")
    print(f"Using MOTION BLURRED images from: {config['image_dir']}\n")

    # Setup Detectron2 with adaptation after res4
    print("[1/5] Setting up Detectron2 with adaptation AFTER res4...")
    model, cfg = setup_detectron2_with_adaptation(
        device=config['device'],
        checkpoint_path=config['checkpoint_path'],
        feature_key=config['feature_key']
    )
    print("✅ Setup complete with adaptation layer after res4\n")

    # DIAGNOSTIC 1: Test identity initialization
    print("[2/5] Running identity initialization test...")
    identity_ok = test_identity_initialization(model, device)
    if not identity_ok:
        print("\n⚠️  WARNING: Identity test failed! Results may be unreliable.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    print()

    # Load energy model
    print("[3/5] Loading trained energy model...")
    try:
        from new_energy_model import ROI_EnergyModel
        energy_channels = 256
        energy_model = ROI_EnergyModel(in_channels=energy_channels).to(device)
        print(f"   Loaded ROI_EnergyModel with {energy_channels} input channels")
    except Exception as e:
        print(f"   ⚠️  Could not import new_energy_model: {e}")
        class DummyEnergyModel(nn.Module):
            def __init__(self, in_channels=256):
                super().__init__()
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(in_channels, 1)
            def forward(self, x):
                n, c, h, w = x.shape
                x = self.pool(x).view(n, c)
                energy = self.fc(x)
                return energy.squeeze(-1)

        energy_model = DummyEnergyModel(in_channels=256).to(device)
        print("   Using fallback DummyEnergyModel")

    if os.path.exists(config['energy_model_path']):
        try:
            checkpoint = torch.load(config['energy_model_path'], map_location=device)
            energy_model.load_state_dict(checkpoint)
            print(f"✅ Loaded energy model from {config['energy_model_path']}\n")
        except Exception as e:
            print(f"⚠️ Failed to load energy model checkpoint: {e}")
            print("   Using randomly initialized energy model\n")
    else:
        print(f"⚠️  Energy model not found at {config['energy_model_path']}")
        print("   Using randomly initialized energy model\n")

    energy_model.eval()
    for param in energy_model.parameters():
        param.requires_grad = False

    # Load dataset
    print("[4/5] Loading dataset...")
    dataset = load_cityscapes_data(
        config['image_dir'],
        config['annotation_file'],
        max_images=config['max_images']
    )

    if len(dataset) == 0:
        print("\n❌ ERROR: No images loaded! Cannot proceed.")
        return

    # DIAGNOSTIC 2: Test pretrained baseline
    test_pretrained_baseline(model, dataset, device)

    # Run sequential adaptation
    print("\n[5/5] Running sequential test-time adaptation...")

    # Count adaptable parameters
    if hasattr(model, 'adaptation_layer') and model.adaptation_layer is not None:
        adaptable_params = [p for p in model.adaptation_layer.parameters() if p.requires_grad]
        total_adapt_params = sum(p.numel() for p in adaptable_params)
    else:
        adaptable_params = []
        total_adapt_params = 0

    print(f"Adaptable parameter tensors: {len(adaptable_params)}")
    print(f"Total adaptable param count: {total_adapt_params:,}")
    print(f"Adapting stage: {model.adaptation_stage}")
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
    output_file = 'sequential_adaptation_results_res4_motion_blur.json'
    with open(output_file, 'w') as f:
        json_results = {k: [float(v) for v in vals] for k, vals in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    main()