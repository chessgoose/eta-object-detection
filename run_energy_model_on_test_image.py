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



def visualize_detections_with_energy(image, instances, pred_energies, save_path="detections_with_energy.png"):
    """
    Overlays detection boxes with colors corresponding to predicted energy levels.
    
    Args:
        image (np.ndarray): BGR image (as loaded by cv2)
        instances (detectron2.structures.Instances): model predictions
        pred_energies (torch.Tensor or np.ndarray): predicted scalar energy per detection
    """
    img = image.copy()
    if isinstance(pred_energies, torch.Tensor):
        pred_energies = pred_energies.detach().cpu().numpy()

    if len(pred_energies.shape) > 1:
        pred_energies = pred_energies.squeeze()

    boxes = instances.pred_boxes.tensor.cpu().numpy()
    scores = instances.scores.cpu().numpy() if instances.has("scores") else None
    classes = instances.pred_classes.cpu().numpy()

    # Normalize energies to [0,1]
    pred_energies = np.clip((pred_energies - pred_energies.min()) / (pred_energies.ptp() + 1e-8), 0, 1)
    cmap = plt.get_cmap("RdYlGn_r")  # red = high energy, green = low energy

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        energy = pred_energies[i]
        color = (np.array(cmap(energy)[:3]) * 255).astype(int).tolist()

        label = f"E={energy:.2f}"
        if scores is not None:
            label += f", S={scores[i]:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Predicted Energies on Detections")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


@torch.no_grad()
def test_energy_model(model_path, clean_dir, blur_dir, ann_file, device="cuda"):
    # Load model and predictor
    predictor = setup_detectron2(device)
    energy_model = ROI_EnergyModel(in_channels=256).to(device)
    energy_model.load_state_dict(torch.load(model_path, map_location=device))
    energy_model.eval()

    # Load validation samples
    paired_data = load_paired_data(clean_dir, blur_dir, ann_file)
    sample = random.choice(paired_data)
    print(f"Testing on: {sample['clean_path']}")

    clean_img = cv2.imread(sample["clean_path"])
    blur_img = cv2.imread(sample["blur_path"])
    H, W = sample["height"], sample["width"]
    gt = prepare_gt_instances(sample["annotations"], (H, W), device)
    print(gt)
    print(H, W)

    clean_out = predictor(clean_img)["instances"].to(device)
    if len(clean_out) == 0 or len(gt) == 0:
        print("No detections or GT boxes found in this image.")
        return

    # Compute detection errors and target energies
    errors = compute_detection_error(clean_out, gt)
    target_E = compute_target_energy(errors, temperature=200)

    # Extract ROI features
    img_tensor = torch.as_tensor(clean_img.astype("float32").transpose(2, 0, 1)).to(device)
    inputs = [{"image": img_tensor}]
    image_list = predictor.model.preprocess_image(inputs)
    roi_feats = extract_roi_features(predictor.model, image_list.tensor, clean_out.pred_boxes.tensor)

    # Forward pass through energy model
    pred_E_map = energy_model(roi_feats).cpu().numpy()  # (N, 1, h, w)
    target_E = target_E.cpu().numpy()

    # Visualize per-ROI energy maps
    num_show = min(4, len(pred_E_map))
    fig, axs = plt.subplots(num_show, 3, figsize=(9, 3 * num_show))

    for i in range(num_show):
        pred_map = pred_E_map[i, 0]
        target_val = target_E[i]
        box = clean_out.pred_boxes.tensor[i].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # Extract ROI image
        roi_img = clean_img[y1:y2, x1:x2, ::-1]  # BGR -> RGB

        axs[i, 0].imshow(roi_img)
        axs[i, 0].set_title(f"ROI {i+1}")
        axs[i, 0].axis("off")

        axs[i, 1].imshow(pred_map, cmap="inferno")
        axs[i, 1].set_title(f"Pred Energy (mean={pred_map.mean():.3f})")
        axs[i, 1].axis("off")

        axs[i, 2].imshow(pred_map, cmap="inferno")
        axs[i, 2].set_title(f"Target Energy = {target_val:.3f}")
        axs[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig("roi_energy_maps.png")
    plt.close()

    # Overlay energy scores on full detections
    pred_E_scalar = np.array([pred_E_map[i, 0].mean() for i in range(len(pred_E_map))])
    visualize_detections_with_energy(clean_img, clean_out, pred_E_scalar, save_path="detections_with_energy.png")

    # Visualize ground truth side-by-side with detections
    visualize_ground_truth(clean_img, gt, save_path="ground_truth.png")

    # Combine both images for comparison
    det_img = plt.imread("detections_with_energy.png")
    gt_img = plt.imread("ground_truth.png")

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    axs[0].imshow(gt_img)
    axs[0].set_title("Ground Truth")
    axs[0].axis("off")

    axs[1].imshow(det_img)
    axs[1].set_title("Predicted Detections + Energies")
    axs[1].axis("off")

    plt.tight_layout()
    plt.savefig("side_by_side_comparison.png")
    plt.close()
    print("✅ Saved ROI energy maps to roi_energy_maps.png")
    print("✅ Saved full detection visualization to detections_with_energy.png")


from detectron2.structures import Instances

def visualize_ground_truth(image, gt_instances, save_path="ground_truth.png"):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Copy ground-truth instances and rename fields
    gt_vis = Instances(gt_instances.image_size)
    gt_vis.pred_boxes = gt_instances.gt_boxes
    gt_vis.pred_classes = gt_instances.gt_classes
    print(gt_vis)
    
    # Visualize using standard Visualizer
    v = Visualizer(img_rgb, scale=1.0)
    v = v.draw_instance_predictions(gt_vis.to("cpu"))
    out = v.get_image()

    plt.figure(figsize=(10, 8))
    plt.imshow(out)
    plt.axis("off")
    plt.title("Ground Truth Annotations (Fixed)")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Example usage
if __name__ == "__main__":
    MODEL_PATH = "multiple_roi_energy_model_epoch2.pth"
    CLEAN_DIR = "/home/lyz6/AMROD/datasets/cityscapes/leftImg8bit/val"
    BLUR_DIR = "/home/lyz6/AMROD/datasets/motion_blur/leftImg8bit/val"
    ANN_FILE = "/home/lyz6/AMROD/datasets/cityscapes/annotations/instancesonly_filtered_gtFine_val.json"

    test_energy_model(MODEL_PATH, CLEAN_DIR, BLUR_DIR, ANN_FILE, device="cuda")