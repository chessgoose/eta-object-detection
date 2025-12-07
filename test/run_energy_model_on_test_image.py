import os, json, glob, random, cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from detectron2.structures import Boxes
from detectron2.utils.visualizer import Visualizer, ColorMode

from new_energy_model import (
    setup_detectron2,
    ROI_EnergyModel,
    load_paired_data,
    prepare_gt_instances,
    compute_detection_error,
    compute_target_energy,
    extract_roi_features,
)
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


# Class names for COCO classes we care about
COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle", 
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck"
}


def draw_text_with_background(img, text, pos, font_scale=0.7, thickness=2, 
                               text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility"""
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, 
                  (x, y - text_height - baseline - 5), 
                  (x + text_width + 5, y + 5),
                  bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return text_height + baseline + 10


def visualize_ground_truth(image, gt_instances, save_path="ground_truth.png"):
    """
    Visualize ground truth bounding boxes with clear labels.
    """
    img = image.copy()
    boxes = gt_instances.gt_boxes.tensor.cpu().numpy()
    classes = gt_instances.gt_classes.cpu().numpy()
    
    # Draw each box with label
    for i, (box, cls) in enumerate(zip(boxes, classes)):
        x1, y1, x2, y2 = map(int, box)
        
        # Draw box in green (ground truth)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Label
        class_name = COCO_CLASS_NAMES.get(cls, f"class_{cls}")
        label = f"GT: {class_name}"
        
        # Draw label with background
        draw_text_with_background(img, label, (x1, y1 - 10), 
                                 font_scale=0.8, thickness=2,
                                 text_color=(255, 255, 255), 
                                 bg_color=(0, 180, 0))
    
    # Add title
    title_text = f"Ground Truth ({len(boxes)} objects)"
    draw_text_with_background(img, title_text, (10, 40),
                             font_scale=1.2, thickness=3,
                             text_color=(255, 255, 255),
                             bg_color=(0, 150, 0))
    
    # Save
    plt.figure(figsize=(16, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_detections_with_energy(image, instances, pred_energies, 
                                     save_path="detections_with_energy.png"):
    """
    Visualize detections with energy-based color coding.
    Red = High energy (unreliable/OOD)
    Green = Low energy (reliable/in-distribution)
    """
    img = image.copy()
    
    if isinstance(pred_energies, torch.Tensor):
        pred_energies = pred_energies.detach().cpu().numpy()
    
    if len(pred_energies.shape) > 1:
        pred_energies = pred_energies.squeeze()
    
    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy() if instances.has("scores") else None
    classes = instances.pred_classes.cpu().numpy()
    
    # Energy color mapping: red (high energy/bad) to green (low energy/good)
    cmap = plt.get_cmap("RdYlGn_r")
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        energy = pred_energies[i]
        
        # Get color based on energy (RGB 0-255)
        color_rgb = (np.array(cmap(energy)[:3]) * 255).astype(int)
        color_bgr = tuple(color_rgb[::-1].tolist())  # Convert RGB to BGR for OpenCV
        
        # Draw thick box
        cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 4)
        
        # Create label
        class_name = COCO_CLASS_NAMES.get(classes[i], f"class_{classes[i]}")
        label = f"{class_name}"
        energy_label = f"Energy: {energy:.3f}"
        
        if scores is not None:
            conf_label = f"Conf: {scores[i]:.2f}"
        else:
            conf_label = ""
        
        # Draw labels with background
        y_offset = y1 - 10
        y_offset -= draw_text_with_background(img, label, (x1, y_offset),
                                             font_scale=0.8, thickness=2,
                                             text_color=(255, 255, 255),
                                             bg_color=color_bgr)
        
        y_offset -= draw_text_with_background(img, energy_label, (x1, y_offset),
                                             font_scale=0.7, thickness=2,
                                             text_color=(255, 255, 255),
                                             bg_color=color_bgr)
        
        if conf_label:
            draw_text_with_background(img, conf_label, (x1, y_offset),
                                     font_scale=0.6, thickness=2,
                                     text_color=(255, 255, 255),
                                     bg_color=color_bgr)
    
    # Add title with legend
    title = f"Predicted Detections ({len(boxes)} objects)"
    draw_text_with_background(img, title, (10, 40),
                             font_scale=1.2, thickness=3,
                             text_color=(255, 255, 255),
                             bg_color=(50, 50, 50))
    
    # Add energy scale legend
    legend_y = 90
    draw_text_with_background(img, "Energy Scale:", (10, legend_y),
                             font_scale=0.9, thickness=2,
                             text_color=(255, 255, 255),
                             bg_color=(50, 50, 50))
    
    legend_y += 40
    draw_text_with_background(img, "GREEN = Low (Reliable)", (10, legend_y),
                             font_scale=0.8, thickness=2,
                             text_color=(255, 255, 255),
                             bg_color=(0, 180, 0))
    
    legend_y += 35
    draw_text_with_background(img, "YELLOW = Medium", (10, legend_y),
                             font_scale=0.8, thickness=2,
                             text_color=(0, 0, 0),
                             bg_color=(0, 255, 255))
    
    legend_y += 35
    draw_text_with_background(img, "RED = High (Unreliable)", (10, legend_y),
                             font_scale=0.8, thickness=2,
                             text_color=(255, 255, 255),
                             bg_color=(0, 0, 255))
    
    # Add statistics
    stats_y = img.shape[0] - 100
    stats = [
        f"Mean Energy: {pred_energies.mean():.3f}",
        f"Min Energy: {pred_energies.min():.3f}",
        f"Max Energy: {pred_energies.max():.3f}"
    ]
    for stat in stats:
        draw_text_with_background(img, stat, (10, stats_y),
                                 font_scale=0.8, thickness=2,
                                 text_color=(255, 255, 255),
                                 bg_color=(50, 50, 50))
        stats_y += 35
    
    # Save
    plt.figure(figsize=(16, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


@torch.no_grad()
def test_energy_model(model_path, clean_dir, blur_dir, ann_file, device="cuda", num_samples=3):
    """
    Test energy model on validation samples and create visualizations.
    
    Args:
        num_samples: Number of random samples to visualize
    """
    # Load model and predictor
    predictor = setup_detectron2(device)
    energy_model = ROI_EnergyModel(in_channels=256).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        energy_model.load_state_dict(checkpoint['model_state_dict'])
    else:
        energy_model.load_state_dict(checkpoint)
    energy_model.eval()

    # Load validation samples
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)
    
    # Process multiple samples
    samples = random.sample(paired_data, min(num_samples, len(paired_data)))
    
    for sample_idx, sample in enumerate(samples):
        print(f"\n{'='*60}")
        print(f"Processing sample {sample_idx + 1}/{len(samples)}")
        print(f"Image: {sample['clean_path']}")
        print(f"{'='*60}")
        
        clean_img = cv2.imread(sample["clean_path"])
        blur_img = cv2.imread(sample["blur_path"])
        H, W = sample["height"], sample["width"]
        gt = prepare_gt_instances(sample["annotations"], (H, W), device)
        
        print(f"Image size: {H} x {W}")
        print(f"Ground truth boxes: {len(gt)}")
        
        # Run detector
        clean_out = predictor(clean_img)["instances"].to(device)
        print(f"Detected boxes: {len(clean_out)}")
        
        if len(clean_out) == 0:
            print("⚠️  No detections found, skipping...")
            continue
        
        if len(gt) == 0:
            print("⚠️  No ground truth boxes, skipping...")
            continue
        
        # Compute detection errors and target energies
        errors = compute_detection_error(clean_out, gt)
        target_E = compute_target_energy(errors, temperature=200)
        
        print(f"Detection errors - Min: {errors.min():.3f}, Max: {errors.max():.3f}, Mean: {errors.mean():.3f}")
        print(f"Target energies - Min: {target_E.min():.3f}, Max: {target_E.max():.3f}, Mean: {target_E.mean():.3f}")
        
        # Extract ROI features
        img_tensor = torch.as_tensor(clean_img.astype("float32").transpose(2, 0, 1)).to(device)
        inputs = [{"image": img_tensor}]
        image_list = predictor.model.preprocess_image(inputs)
        roi_feats = extract_roi_features(predictor.model, image_list.tensor, clean_out.pred_boxes.tensor)
        
        # Forward pass through energy model
        pred_E_map = energy_model(roi_feats)  # (N, 1, h, w)
        pred_E_scalar = pred_E_map.mean(dim=(2, 3)).squeeze().cpu().numpy()  # Average spatial dims
        
        print(f"Predicted energies - Min: {pred_E_scalar.min():.3f}, Max: {pred_E_scalar.max():.3f}, Mean: {pred_E_scalar.mean():.3f}")
        
        # Create output directory
        output_dir = f"visualization_sample_{sample_idx + 1}"
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Visualize ground truth
        gt_path = os.path.join(output_dir, "1_ground_truth.png")
        visualize_ground_truth(clean_img, gt, save_path=gt_path)
        print(f"✅ Saved ground truth to {gt_path}")
        
        # 2. Visualize detections with energy
        det_path = os.path.join(output_dir, "2_detections_with_energy.png")
        visualize_detections_with_energy(clean_img, clean_out, pred_E_scalar, save_path=det_path)
        print(f"✅ Saved detections with energy to {det_path}")
        
        # 3. Create side-by-side comparison
        gt_img = plt.imread(gt_path)
        det_img = plt.imread(det_path)
        
        fig, axs = plt.subplots(1, 2, figsize=(32, 10))
        axs[0].imshow(gt_img)
        axs[0].set_title("Ground Truth Annotations", fontsize=20, fontweight='bold')
        axs[0].axis("off")
        
        axs[1].imshow(det_img)
        axs[1].set_title("Predicted Detections with Energy Scores", fontsize=20, fontweight='bold')
        axs[1].axis("off")
        
        plt.tight_layout()
        comparison_path = os.path.join(output_dir, "3_side_by_side_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved side-by-side comparison to {comparison_path}")
        
        # 4. Visualize individual ROI energy maps (top 6 highest/lowest energy)
        pred_E_map_cpu = pred_E_map.cpu().numpy()
        target_E_cpu = target_E.cpu().numpy()
        
        # Sort by predicted energy
        sorted_indices = np.argsort(pred_E_scalar)
        
        # Show 3 lowest and 3 highest energy detections
        num_show = min(6, len(sorted_indices))
        indices_to_show = []
        
        # Lowest energy (most reliable)
        indices_to_show.extend(sorted_indices[:min(3, num_show)])
        
        # Highest energy (least reliable)
        if num_show > 3:
            indices_to_show.extend(sorted_indices[-min(3, num_show-3):])
        
        fig, axs = plt.subplots(len(indices_to_show), 3, figsize=(12, 4 * len(indices_to_show)))
        if len(indices_to_show) == 1:
            axs = axs.reshape(1, -1)
        
        for plot_idx, det_idx in enumerate(indices_to_show):
            pred_map = pred_E_map_cpu[det_idx, 0]
            target_val = target_E_cpu[det_idx]
            pred_val = pred_E_scalar[det_idx]
            box = clean_out.pred_boxes.tensor[det_idx].cpu().numpy().astype(int)
            x1, y1, x2, y2 = box
            
            # Extract ROI image
            roi_img = clean_img[max(0, y1):min(H, y2), max(0, x1):min(W, x2), ::-1]  # BGR -> RGB
            
            # Column 1: ROI image
            axs[plot_idx, 0].imshow(roi_img)
            axs[plot_idx, 0].set_title(f"Detection #{det_idx+1}\nBox: [{x1},{y1},{x2},{y2}]", 
                                      fontsize=10, fontweight='bold')
            axs[plot_idx, 0].axis("off")
            
            # Column 2: Predicted energy map
            im = axs[plot_idx, 1].imshow(pred_map, cmap="RdYlGn_r", vmin=0, vmax=1)
            axs[plot_idx, 1].set_title(f"Predicted Energy Map\nMean: {pred_val:.3f}", 
                                      fontsize=10, fontweight='bold')
            axs[plot_idx, 1].axis("off")
            plt.colorbar(im, ax=axs[plot_idx, 1], fraction=0.046, pad=0.04)
            
            # Column 3: Target energy (uniform)
            target_map = np.ones_like(pred_map) * target_val
            im = axs[plot_idx, 2].imshow(target_map, cmap="RdYlGn_r", vmin=0, vmax=1)
            axs[plot_idx, 2].set_title(f"Target Energy\nValue: {target_val:.3f}", 
                                      fontsize=10, fontweight='bold')
            axs[plot_idx, 2].axis("off")
            plt.colorbar(im, ax=axs[plot_idx, 2], fraction=0.046, pad=0.04)
        
        plt.suptitle(f"ROI Energy Analysis (Sorted by Reliability)", 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        roi_path = os.path.join(output_dir, "4_roi_energy_maps.png")
        plt.savefig(roi_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved ROI energy maps to {roi_path}")
        
        print(f"\n🎉 All visualizations saved to: {output_dir}/")


# Example usage
if __name__ == "__main__":
    MODEL_PATH = "models/multiple_roi_energy_model.pth"
    CLEAN_DIR = "/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/val"
    BLUR_DIR = "/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val"
    ANN_FILE = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"

    test_energy_model(
        MODEL_PATH, 
        CLEAN_DIR, 
        BLUR_DIR, 
        ANN_FILE, 
        device="cuda",
        num_samples=3  # Process 3 random samples
    )