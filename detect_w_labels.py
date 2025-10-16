#!/usr/bin/env python3
"""
Detectron2 Class Label Conversion Guide
Shows multiple ways to convert class IDs to actual class names
"""

import os
import sys
import torch
import cv2
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog


# Method 1: Using MetadataCatalog (Recommended)
def get_class_names_from_metadata(cfg):
    """
    Get class names using MetadataCatalog
    
    Args:
        cfg: Detectron2 config object
        
    Returns:
        List of class names
    """
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    class_names = metadata.thing_classes  # For detection/instance segmentation
    return class_names


# Method 2: Manual COCO class names (if metadata not available)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def convert_predictions_to_labels(outputs, class_names):
    """
    Convert detection outputs to readable format with class names
    
    Args:
        outputs: Detectron2 prediction outputs
        class_names: List of class names
        
    Returns:
        List of dictionaries with detection info
    """
    instances = outputs["instances"].to("cpu")
    
    detections = []
    for i in range(len(instances)):
        detection = {
            'class_id': instances.pred_classes[i].item(),
            'class_name': class_names[instances.pred_classes[i].item()],
            'confidence': instances.scores[i].item(),
            'bbox': instances.pred_boxes[i].tensor.numpy()[0].tolist(),
        }
        detections.append(detection)
    
    return detections


def print_detections_with_labels(detections):
    """
    Print detections in a readable format
    
    Args:
        detections: List of detection dictionaries
    """
    print("\n" + "="*80)
    print(f"FOUND {len(detections)} OBJECTS")
    print("="*80)
    
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        width = x2 - x1
        height = y2 - y1
        
        print(f"\nObject {i+1}:")
        print(f"  Class: {det['class_name'].upper()}")
        print(f"  Confidence: {det['confidence']:.2%}")
        print(f"  Bounding Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
        print(f"  Size: {width:.1f} x {height:.1f} pixels")
    
    print("\n" + "="*80)


def get_detection_summary(detections):
    """
    Get summary statistics of detections
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Dictionary with summary statistics
    """
    if not detections:
        return {"total": 0, "by_class": {}}
    
    # Count objects by class
    class_counts = {}
    for det in detections:
        class_name = det['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    # Sort by count
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "total": len(detections),
        "by_class": dict(sorted_classes),
        "unique_classes": len(class_counts)
    }


def print_summary(summary):
    """
    Print detection summary
    
    Args:
        summary: Summary dictionary
    """
    print("\n" + "="*80)
    print("DETECTION SUMMARY")
    print("="*80)
    print(f"\nTotal Objects Detected: {summary['total']}")
    print(f"Unique Classes: {summary['unique_classes']}")
    
    if summary['by_class']:
        print(f"\nBreakdown by Class:")
        for class_name, count in summary['by_class'].items():
            print(f"  • {class_name}: {count}")
    
    print("="*80)


class LabeledObjectDetector:
    """Object detector with automatic label conversion"""
    
    def __init__(self, model_name="COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
                 confidence_threshold=0.5):
        """
        Initialize detector
        
        Args:
            model_name: Model configuration name
            confidence_threshold: Minimum confidence for detections
        """
        print("Initializing Labeled Object Detector...")
        
        # Setup config
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_name)
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create predictor
        self.predictor = DefaultPredictor(self.cfg)
        
        # Get class names
        try:
            self.class_names = get_class_names_from_metadata(self.cfg)
            print(f"✓ Loaded {len(self.class_names)} class names from metadata")
        except:
            self.class_names = COCO_CLASSES
            print(f"✓ Using default COCO class names ({len(self.class_names)} classes)")
        
        print(f"✓ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print()
    
    def detect(self, image_path):
        """
        Detect objects and return with labels
        
        Args:
            image_path: Path to image
            
        Returns:
            image: Original image
            detections: List of labeled detections
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        print(f"Processing: {image_path}")
        print(f"  Image size: {image.shape[1]}x{image.shape[0]}")
        
        # Run detection
        outputs = self.predictor(image)
        
        # Convert to labeled format
        detections = convert_predictions_to_labels(outputs, self.class_names)
        
        print(f"  Detected: {len(detections)} objects")
        
        return image, detections
    
    def get_class_name(self, class_id):
        """
        Get class name from class ID
        
        Args:
            class_id: Integer class ID
            
        Returns:
            Class name string
        """
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        else:
            return f"unknown_class_{class_id}"
    
    def filter_by_class(self, detections, class_names):
        """
        Filter detections by class name(s)
        
        Args:
            detections: List of detections
            class_names: String or list of class names to keep
            
        Returns:
            Filtered list of detections
        """
        if isinstance(class_names, str):
            class_names = [class_names]
        
        class_names_lower = [cn.lower() for cn in class_names]
        filtered = [d for d in detections if d['class_name'].lower() in class_names_lower]
        
        return filtered
    
    def get_most_confident(self, detections, top_k=5):
        """
        Get top-k most confident detections
        
        Args:
            detections: List of detections
            top_k: Number of top detections to return
            
        Returns:
            List of top-k detections
        """
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        return sorted_detections[:top_k]


def example_usage():
    """Example of how to use label conversion"""
    
    print("="*80)
    print("DETECTRON2 CLASS LABEL CONVERSION EXAMPLE")
    print("="*80)
    print()
    
    # Check for input image
    image_path = "/home/lyz6/palmer_scratch/eta-object-detection/PhotoFunia-1759344129.jpg"
    
    # Create detector
    detector = LabeledObjectDetector(confidence_threshold=0.5)
    
    # Run detection
    image, detections = detector.detect(image_path)
    
    # Print all detections with labels
    print_detections_with_labels(detections)
    
    # Print summary
    summary = get_detection_summary(detections)
    print_summary(summary)
    
    # Example: Filter by class
    print("\n" + "="*80)
    print("FILTERING EXAMPLES")
    print("="*80)
    
    # Get only people
    people = detector.filter_by_class(detections, 'person')
    print(f"\nFound {len(people)} person(s)")
    
    # Get vehicles (car, truck, bus)
    vehicles = detector.filter_by_class(detections, ['car', 'truck', 'bus'])
    print(f"Found {len(vehicles)} vehicle(s)")
    
    # Get top 3 most confident detections
    top_3 = detector.get_most_confident(detections, top_k=3)
    print(f"\nTop 3 most confident detections:")
    for i, det in enumerate(top_3, 1):
        print(f"  {i}. {det['class_name']} ({det['confidence']:.2%})")
    
    # Example: Access individual detection properties
    if detections:
        print("\n" + "="*80)
        print("EXAMPLE: Accessing Detection Properties")
        print("="*80)
        
        first_det = detections[0]
        print(f"\nFirst detection:")
        print(f"  Class ID: {first_det['class_id']}")
        print(f"  Class Name: {first_det['class_name']}")
        print(f"  Confidence: {first_det['confidence']:.4f}")
        print(f"  Bounding Box (x1, y1, x2, y2): {first_det['bbox']}")
        
        # Calculate center point
        x1, y1, x2, y2 = first_det['bbox']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        print(f"  Center Point: ({center_x:.1f}, {center_y:.1f})")
    
    print("\n" + "="*80)
    print("✓ EXAMPLE COMPLETED")
    print("="*80)


if __name__ == "__main__":
    example_usage()