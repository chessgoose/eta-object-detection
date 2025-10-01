#!/usr/bin/env python3
"""
Detectron2 Object Detection Demo Script
Runs a pre-trained object detection model on a sample image
"""

import torch
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt
from pathlib import Path


def setup_cfg(model_name="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml", 
              confidence_threshold=0.5):
    """
    Set up detectron2 config for inference
    
    Args:
        model_name: Model configuration file from model zoo
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        cfg: Detectron2 config object
    """
    cfg = get_cfg()
    # Load model config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file(model_name))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    # Download and use pre-trained weights
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
    # Set device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    return cfg


def load_image(image_path):
    """
    Load an image from file
    
    Args:
        image_path: Path to image file
    
    Returns:
        image: Image in BGR format (OpenCV convention)
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    return image


def run_detection(predictor, image):
    """
    Run object detection on an image
    
    Args:
        predictor: Detectron2 predictor
        image: Input image in BGR format
    
    Returns:
        outputs: Detection results containing boxes, classes, scores
    """
    outputs = predictor(image)
    return outputs


def visualize_predictions(image, outputs, metadata):
    """
    Visualize detection results on the image
    
    Args:
        image: Original image in BGR format
        outputs: Detection outputs from predictor
        metadata: Metadata catalog for the dataset
    
    Returns:
        vis_image: Visualized image with detections
    """
    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    vis_image = out.get_image()[:, :, ::-1]
    return vis_image


def print_detection_summary(outputs):
    """
    Print summary of detection results
    
    Args:
        outputs: Detection outputs from predictor
    """
    instances = outputs["instances"].to("cpu")
    num_detections = len(instances)
    
    print(f"\nDetection Summary:")
    print(f"Number of detections: {num_detections}")
    
    if num_detections > 0:
        print(f"\nDetailed Results:")
        for i in range(num_detections):
            bbox = instances.pred_boxes[i].tensor.numpy()[0]
            score = instances.scores[i].item()
            class_id = instances.pred_classes[i].item()
            
            print(f"\nDetection {i+1}:")
            print(f"  Class ID: {class_id}")
            print(f"  Confidence: {score:.3f}")
            print(f"  Bounding Box: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")


def save_results(original_image, visualized_image, output_path):
    """
    Save visualization results
    
    Args:
        original_image: Original input image
        visualized_image: Image with detection visualizations
        output_path: Path to save the output image
    """
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image", fontsize=16)
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Detections", fontsize=16)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    plt.close()


def main():
    """
    Main function to run object detection demo
    """
    print("=" * 60)
    print("Detectron2 Object Detection Demo")
    print("=" * 60)
    
    # Configuration
    image_path = "/home/lyz6/cityscapes/test/berlin/berlin_000000_000019_leftImg8bit.png"  # Change this to your image path
    output_path = "detection_results.jpg"
    model_name = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    confidence_threshold = 0.5
    
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Confidence Threshold: {confidence_threshold}")
    print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    # Setup configuration
    print("\n[1/5] Setting up configuration...")
    cfg = setup_cfg(model_name, confidence_threshold)
    
    # Create predictor
    print("[2/5] Loading model...")
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Load image
    print(f"[3/5] Loading image from {image_path}...")
    try:
        image = load_image(image_path)
        print(f"  Image shape: {image.shape}")
    except ValueError as e:
        print(f"Error: {e}")
        print("\nTip: Download a sample image with:")
        print("  wget http://images.cocodataset.org/val2017/000000439715.jpg -O input_image.jpg")
        return
    
    # Run detection
    print("[4/5] Running object detection...")
    outputs = run_detection(predictor, image)
    
    # Print results
    print_detection_summary(outputs)
    
    # Visualize and save
    print("\n[5/5] Visualizing and saving results...")
    vis_image = visualize_predictions(image, outputs, metadata)
    save_results(image, vis_image, output_path)
    
    print("\n" + "=" * 60)
    print("Detection complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()