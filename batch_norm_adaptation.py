"""
Detectron2 Sequential Test-Time Adaptation with Batch Normalization Adaptation
BN Adaptation: Updates batch normalization statistics at test time to adapt to target domain
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

from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.structures import Instances
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.layers import FrozenBatchNorm2d

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
            momentum=0.1,  # Use standard momentum for adaptation
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


def count_bn_layers(module):
    """Count the number of BatchNorm layers in a module."""
    count = 0
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, FrozenBatchNorm2d)):
            count += 1
    return count


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
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
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
# BN ADAPTATION UTILITIES
# ==============================================================

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


def reset_bn_stats(model):
    """Reset BN running statistics to initial values."""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.track_running_stats:
                module.running_mean.zero_()
                module.running_var.fill_(1)
                module.num_batches_tracked.zero_()


# ==============================================================
# SEQUENTIAL TEST-TIME ADAPTATION WITH BN ADAPTATION
# ==============================================================

def sequential_bn_adaptation(
    model,
    dataset,
    device,
    adaptation_lr=0.001,
    iterations_per_image=1,
    score_threshold=0.05,
    update_affine=True,
    verbose=False
):
    """
    Sequential test-time adaptation using BN adaptation.
    
    Args:
        model: Detectron2 model with BN layers
        dataset: List of image data dicts
        device: Device to run on
        adaptation_lr: Learning rate for BN affine parameter updates
        iterations_per_image: Number of forward passes per image for adaptation
        score_threshold: Score threshold for predictions
        update_affine: Whether to update BN affine parameters or just statistics
        verbose: Whether to print detailed info
    """
    # Setup optimizer for BN affine parameters if updating them
    if update_affine:
        bn_params = []
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if module.weight is not None:
                    bn_params.append(module.weight)
                if module.bias is not None:
                    bn_params.append(module.bias)
        
        optimizer = optim.SGD(bn_params, lr=adaptation_lr, momentum=0.9)
        print(f"Optimizing {len(bn_params)} BN affine parameters")
    else:
        optimizer = None
        print("Only updating BN statistics (not affine parameters)")
    
    results = {
        'map_before': [],
        'map_after': [],
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
        
        # Prepare image tensor
        height, width = image.shape[:2]
        image_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1)).to(device)
        
        # Evaluate BEFORE adaptation (with eval mode BN)
        model.eval()
        set_bn_to_eval(model)
        
        with torch.no_grad():
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            outputs = model(inputs)
            predictions_before = outputs[0]['instances']
        
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
        
        # Perform BN adaptation iterations
        model.eval()  # Keep model in eval mode
        set_bn_to_train(model)  # But set BN to training mode for statistics update
        
        for iter_idx in range(iterations_per_image):
            inputs = [{"image": image_tensor, "height": height, "width": width}]
            
            if update_affine and optimizer:
                # If updating affine parameters, we need gradients
                optimizer.zero_grad()
                
                # Enable gradients for this forward pass
                with torch.enable_grad():
                    # Preprocess image
                    images = model.preprocess_image(inputs)
                    
                    # Forward through backbone to get features
                    features = model.backbone(images.tensor)
                    
                    # Forward through proposal generator to get proposals
                    proposals, proposal_losses = model.proposal_generator(images, features, None)
                    
                    # Forward through ROI heads to get predictions
                    detector_results, detector_losses = model.roi_heads(images, features, proposals, None)
                    
                    # Compute entropy minimization loss
                    # We want the model to make confident predictions on test data
                    loss = 0
                    num_instances = 0
                    
                    for result in detector_results:
                        if len(result) > 0:
                            # Get class scores (before softmax if available, or use final scores)
                            if hasattr(result, 'scores'):
                                scores = result.scores
                                # Entropy minimization: maximize confidence
                                # = minimize negative log probability
                                # = minimize -log(score)
                                loss += -(torch.log(scores + 1e-6)).mean()
                                num_instances += len(scores)
                    
                    # If we have predictions, optimize
                    if num_instances > 0 and torch.isfinite(loss):
                        loss.backward()
                        optimizer.step()
            else:
                # Just forward pass to update statistics
                with torch.no_grad():
                    outputs = model(inputs)
        
        # Evaluate AFTER adaptation (with eval mode BN using updated statistics)
        model.eval()
        set_bn_to_eval(model)
        
        with torch.no_grad():
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
        results['num_detections_before'].append(len(pred_before['boxes']))
        results['num_detections_after'].append(len(pred_after['boxes']))
        
        # Print results
        map_delta = map_after - map_before
        print(f"  📊 mAP: {map_before:.3f} → {map_after:.3f} (Δ: {map_delta:+.3f})")
        print(f"  📦 Detections: {results['num_detections_before'][-1]} → {results['num_detections_after'][-1]}")
        print()
    
    # Print final summary
    print("=" * 60)
    print("Final Results")
    print("=" * 60)
    print(f"Average mAP before:  {np.mean(results['map_before']):.4f}")
    print(f"Average mAP after:   {np.mean(results['map_after']):.4f}")
    print(f"Average improvement: {np.mean(results['map_after']) - np.mean(results['map_before']):+.4f}")
    print("=" * 60)
    
    return results


# ==============================================================
# MAIN
# ==============================================================

def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("BN Adaptation for Object Detection - Motion Blurred Images")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        # MOTION BLURRED IMAGES DIRECTORY
        'image_dir': '/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val',
        'annotation_file': '/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        
        # Optional: Load from previous checkpoint instead of COCO weights
        'checkpoint_path': None,
        
        'max_images': 50,
        'adaptation_lr': 0.01,  # Learning rate for BN affine parameters
        'iterations_per_image': 3,  # Usually 1-3 iterations
        'score_threshold': 0.05,
        
        # BN Adaptation mode:
        # False: Only update BN statistics (simple baseline)
        # True: Update BN affine parameters via entropy minimization (standard TENT approach)
        'update_affine': True,
        
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    
    device = torch.device(config['device'])
    print(f"Using device: {device}")
    print(f"Using MOTION BLURRED images from: {config['image_dir']}\n")
    
    # Setup Detectron2 with BN adaptation
    print("[1/3] Setting up Detectron2 with BN adaptation...")
    model, cfg = setup_detectron2_for_bn_adaptation(
        device,
        checkpoint_path=config['checkpoint_path']
    )
    print("✅ Setup complete with BN layers ready for adaptation\n")
    
    # Load dataset
    print("[2/3] Loading dataset...")
    dataset = load_cityscapes_data(
        config['image_dir'],
        config['annotation_file'],
        max_images=config['max_images']
    )
    
    if len(dataset) == 0:
        print("\n❌ ERROR: No images loaded! Cannot proceed.")
        return
    
    print()
    
    # Run sequential BN adaptation
    print("[3/3] Running sequential test-time BN adaptation...")
    print(f"Update affine parameters: {config['update_affine']}")
    print(f"Adaptation LR: {config['adaptation_lr']}")
    print(f"Iterations per image: {config['iterations_per_image']}")
    print()
    
    results = sequential_bn_adaptation(
        model=model,
        dataset=dataset,
        device=device,
        adaptation_lr=config['adaptation_lr'],
        iterations_per_image=config['iterations_per_image'],
        score_threshold=config['score_threshold'],
        update_affine=config['update_affine'],
        verbose=True
    )
    
    # Save results
    output_file = 'bn_adaptation_results_motion_blur.json'
    with open(output_file, 'w') as f:
        json_results = {k: [float(v) for v in vals] for k, vals in results.items()}
        json.dump(json_results, f, indent=2)
    print(f"\n✅ Results saved to {output_file}")


if __name__ == '__main__':
    main()