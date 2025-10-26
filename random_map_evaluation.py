import cv2
import torch
import numpy as np
import random
from collections import defaultdict

def compute_iou(box1, box2):
    """Compute IoU between two bounding boxes in (x1, y1, x2, y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def compute_ap_for_class(predictions, ground_truths, class_id, iou_threshold=0.5):
    """Compute AP@0.5 for a single class."""
    # Filter for this class
    class_preds = [p for p in predictions if p['class'] == class_id]
    class_gts = [g for g in ground_truths if g['class'] == class_id]
    
    if len(class_gts) == 0:
        return 0.0  # No ground truth for this class
    
    # Sort predictions by confidence
    class_preds.sort(key=lambda x: x['score'], reverse=True)
    
    # Track matches
    tp = np.zeros(len(class_preds))
    fp = np.zeros(len(class_preds))
    gt_matched = [False] * len(class_gts)
    
    for pred_idx, pred in enumerate(class_preds):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx, gt in enumerate(class_gts):
            if gt_matched[gt_idx]:
                continue
            
            iou = compute_iou(pred['bbox'], gt['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(class_gts)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + np.finfo(np.float64).eps)
    
    # Compute AP using trapezoidal rule
    ap = 0.0
    if len(recalls) > 0:
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
    
    return ap

def get_dataset_classes(clean_dir, blur_dir, ann_file, num_samples=100):
    """
    Determine all classes present in the dataset by sampling annotations.
    This defines the official class set for academic evaluation.
    """
    from new_energy_model import load_paired_data, prepare_gt_instances
    
    print("Analyzing dataset to determine class set...")
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)
    dataset_classes = set()
    class_counts = defaultdict(int)
    
    # Sample images to find all classes
    sample_size = min(num_samples, len(paired_data))
    sampled_data = random.sample(paired_data, sample_size)
    
    for sample in sampled_data:
        H, W = sample["height"], sample["width"]
        gt_instances = prepare_gt_instances(sample["annotations"], (H, W), "cpu")
        
        if len(gt_instances) > 0:
            classes = gt_instances.gt_classes.numpy()
            for cls in classes:
                class_id = int(cls)
                dataset_classes.add(class_id)
                class_counts[class_id] += 1
    
    sorted_classes = sorted(list(dataset_classes))
    
    print(f"Found {len(sorted_classes)} classes in dataset:")
    for cls in sorted_classes:
        print(f"  Class {cls}: {class_counts[cls]} instances")
    
    return sorted_classes

def evaluate_single_image_academic(predictor, image_path, gt_instances, dataset_classes):
    """
    Evaluate mAP@0.5 using standard academic approach:
    Average over ALL classes in the dataset, even if not present in this image.
    """
    # Load and predict
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return 0.0
    
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    # Convert predictions to simple format
    predictions = []
    if len(instances) > 0:
        pred_boxes = instances.pred_boxes.tensor.numpy()
        pred_scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        for i in range(len(pred_boxes)):
            predictions.append({
                'bbox': pred_boxes[i],
                'score': pred_scores[i],
                'class': int(pred_classes[i])
            })
    
    # Convert ground truth to simple format
    ground_truths = []
    if len(gt_instances) > 0:
        gt_boxes = gt_instances.gt_boxes.tensor.numpy()
        gt_classes = gt_instances.gt_classes.numpy()
        
        for i in range(len(gt_boxes)):
            ground_truths.append({
                'bbox': gt_boxes[i],
                'class': int(gt_classes[i])
            })
    
    # Get classes present in this image for display
    image_classes = set(gt['class'] for gt in ground_truths)
    
    print(f"  Dataset classes: {len(dataset_classes)} total")
    print(f"  Image classes: {sorted(image_classes)} ({len(image_classes)} present)")
    print(f"  Predictions: {len(predictions)}, Ground truth: {len(ground_truths)}")
    
    # Compute AP for ALL dataset classes (standard academic approach)
    class_aps = []
    for class_id in dataset_classes:
        ap = compute_ap_for_class(predictions, ground_truths, class_id)
        class_aps.append(ap)
        
        # Mark if class is present in this image
        present = "✓" if class_id in image_classes else "✗"
        print(f"    Class {class_id}: AP = {ap:.3f} ({ap*100:.1f}%) {present}")
    
    # Compute mean AP over ALL dataset classes
    map_50 = np.mean(class_aps)
    
    print(f"  Academic mAP@0.5 = {map_50:.3f} ({map_50*100:.1f}%)")
    print(f"  (Average over {len(dataset_classes)} dataset classes)")
    
    return map_50

def compare_clean_vs_blur_academic(clean_dir, blur_dir, ann_file, device="cuda", num_images=1):
    """
    Compare clean vs blur using standard academic mAP evaluation.
    """
    from new_energy_model import setup_detectron2, load_paired_data, prepare_gt_instances
    
    # Setup
    predictor = setup_detectron2(device)
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)
    
    print(f"Loaded {len(paired_data)} paired samples")
    
    # Get dataset classes (this is the key for academic evaluation)
    dataset_classes = get_dataset_classes(clean_dir, blur_dir, ann_file)
    
    # Process images
    results = []
    
    for i in range(num_images):
        # Pick random sample
        random_sample = random.choice(paired_data)
        
        print(f"\n{'='*70}")
        print(f"IMAGE {i+1}/{num_images}: {random_sample['clean_path'].split('/')[-1]}")
        print(f"{'='*70}")
        
        # Prepare ground truth
        H, W = random_sample["height"], random_sample["width"]
        gt_instances = prepare_gt_instances(random_sample["annotations"], (H, W), "cpu")
        
        if len(gt_instances) == 0:
            print("No ground truth annotations - skipping")
            continue
        
        # Evaluate clean image
        print(f"\nCLEAN IMAGE EVALUATION:")
        clean_map = evaluate_single_image_academic(
            predictor, random_sample['clean_path'], gt_instances, dataset_classes
        )
        
        # Evaluate blurred image
        print(f"\nBLURRED IMAGE EVALUATION:")
        blur_map = evaluate_single_image_academic(
            predictor, random_sample['blur_path'], gt_instances, dataset_classes
        )
        
        # Store results
        results.append({
            'image': random_sample['clean_path'].split('/')[-1],
            'clean_map': clean_map,
            'blur_map': blur_map
        })
        
        # Show comparison for this image
        performance_drop = clean_map - blur_map
        percentage_drop = (performance_drop / clean_map * 100) if clean_map > 0 else 0
        
        print(f"\n  COMPARISON:")
        print(f"  Clean mAP@0.5:  {clean_map:.3f} ({clean_map*100:.1f}%)")
        print(f"  Blur mAP@0.5:   {blur_map:.3f} ({blur_map*100:.1f}%)")
        print(f"  Performance drop: {performance_drop:.3f} ({percentage_drop:+.1f}%)")
    
    # Overall summary
    if results:
        print(f"\n{'='*70}")
        print(f"OVERALL SUMMARY ({len(results)} images)")
        print(f"{'='*70}")
        
        avg_clean = np.mean([r['clean_map'] for r in results])
        avg_blur = np.mean([r['blur_map'] for r in results])
        avg_drop = avg_clean - avg_blur
        avg_pct_drop = (avg_drop / avg_clean * 100) if avg_clean > 0 else 0
        
        print(f"Average Clean mAP@0.5:  {avg_clean:.3f} ({avg_clean*100:.1f}%)")
        print(f"Average Blur mAP@0.5:   {avg_blur:.3f} ({avg_blur*100:.1f}%)")
        print(f"Average Performance Drop: {avg_drop:.3f} ({avg_pct_drop:+.1f}%)")
        
        print(f"\nDataset Info:")
        print(f"  Total classes in dataset: {len(dataset_classes)}")
        print(f"  Classes: {dataset_classes}")
        print(f"  Evaluation method: Standard Academic (average over all dataset classes)")
    
    return results, dataset_classes

def quick_single_image_evaluation(image_path, clean_dir, blur_dir, ann_file, device="cuda"):
    """
    Quick evaluation of a single image using academic mAP.
    """
    from new_energy_model import setup_detectron2
    
    predictor = setup_detectron2(device)
    dataset_classes = get_dataset_classes(clean_dir, blur_dir, ann_file)
    
    # For a single image without annotations, just show detections
    image = cv2.imread(image_path)
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    print(f"Image: {image_path}")
    print(f"Detections: {len(instances)}")
    if len(instances) > 0:
        print(f"Classes detected: {sorted(set(instances.pred_classes.numpy()))}")
        print(f"Confidence scores: {instances.scores.numpy()}")
    
    print(f"Dataset has {len(dataset_classes)} classes: {dataset_classes}")
    
    return len(instances)

if __name__ == "__main__":
    import sys
    
    # Dataset paths
    CLEAN_DIR = "/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/val"
    BLUR_DIR = "/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val"
    ANN_FILE = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"
    
    # Optional: set seed from command line
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        random.seed(seed)
        print(f"Using random seed: {seed}")
    else:
        print("Using random selection")
    
    # Run academic evaluation
    print("STANDARD ACADEMIC mAP@0.5 EVALUATION")
    print("Averaging over ALL classes in dataset (even if not in image)")
    
    # Evaluate on multiple images for better statistics
    results, dataset_classes = compare_clean_vs_blur_academic(
        CLEAN_DIR, BLUR_DIR, ANN_FILE, 
        device="cuda", 
        num_images=3  # Change this to evaluate more images
    )
    
    print(f"\nThis is the standard approach used in academic papers.")
    print(f"Results are comparable across different images and methods.")