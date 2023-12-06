import os
import shutil
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from configs import parser
from model.contrast.model_main import MainModel
from loaders.get_loader import load_all_imgs, get_transform

shutil.rmtree("vis_batch_adjust_with_ep/", ignore_errors=True)
os.makedirs("vis_batch_adjust_with_ep/", exist_ok=True)


def load_masks(root_dir, imgs_val, labels_val, cat, num_cpts):
    masks = []

    for i, img_path in enumerate(tqdm(imgs_val, desc="Loading masks")):
        # Get the category label for the current image
        label_index = labels_val[i]
        category_name = cat[label_index]
        img_masks = []

        for cpt_num in range(num_cpts):
            mask_path = os.path.join(root_dir, f"cpt{cpt_num}", category_name, f'mask_{i}_{category_name}.png')
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                img_masks.append(mask_img)
            else:
                print(f"Warning: Missing mask at {mask_path}")

        masks.append(img_masks)

    return masks


def load_expert_masks(root_dir, num_cpts, categories):
    expert_masks = {}
    for cpt_num in range(num_cpts):
        expert_masks[cpt_num] = {}
        for category in categories:
            mask_path = os.path.join(root_dir, f"cpt{cpt_num}", category, f'weighted_expert_mask_{category}.png')
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert('L')
                expert_masks[cpt_num][category] = mask_img
            else:
                print(f"Warning: Missing expert mask at {mask_path}")
    return expert_masks


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def process_image(img, mask, threshold, keep_masked_area=True):
    mask = np.array(mask) > threshold
    img_array = np.array(img)
    if keep_masked_area:
        img_array[~mask] = 0
    else:
        img_array[mask] = 0
    return Image.fromarray(img_array)


def evaluate_accuracy(predictions, labels):
    correct = sum(p == l for p, l in zip(predictions, labels))
    return correct / len(predictions)


def main():
    # Load all images
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    transform = get_transform(args)["val"]

    # Load model and weights
    model = MainModel(args)
    device = torch.device("cuda:0")
    model.to(device)
    checkpoint_path = os.path.join(
        args.output_dir,
        f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_"
        f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    num_cpts = args.num_cpt
    # Load masks
    masks = load_masks("vis_val_masks", imgs_val, labels_val, cat, num_cpts)
    # Load expert masks
    expert_masks = load_expert_masks("vis_pp", num_cpts, cat)
    
    # Variables to track predictions
    original_predictions = []
    processed_predictions_masked = []
    processed_predictions_unmasked = []
    
    # Threshold for mask processing
    mask_threshold = int(input("Enter the mask intensity threshold value (0-255): "))
    
    # Initialize CPT error counts for each category
    cpt_error_counts = np.zeros((len(cat), num_cpts), dtype=int)
    
    # Evaluate model
    for i, img_path in enumerate(tqdm(imgs_val, desc="Evaluating model")):
        img_orl = Image.open(img_path).convert('RGB')
        cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device))
        pred_label = pred.argmax(dim=1).item()
        original_predictions.append(pred_label)

        if pred_label != labels_val[i]:  # Misclassified sample
            ious = []
            for cpt_num in range(num_cpts):
                iou = calculate_iou(np.array(masks[i][cpt_num]), np.array(expert_masks[cpt_num][cat[labels_val[i]]]))
                ious.append(iou)

            min_iou_cpt = ious.index(min(ious))
            min_iou_mask = expert_masks[min_iou_cpt][cat[labels_val[i]]]
            
            cpt_error_counts[labels_val[i]][min_iou_cpt] += 1

            # Process image with and without mask
            img_masked = process_image(img_orl, min_iou_mask, mask_threshold, True)
            img_unmasked = process_image(img_orl, min_iou_mask, mask_threshold, False)

            # Predict with processed images
            _, pred_masked, _, _ = model(transform(img_masked).unsqueeze(0).to(device))
            pred_masked_label = pred_masked.argmax(dim=1).item()
            _, pred_unmasked, _, _ = model(transform(img_unmasked).unsqueeze(0).to(device))
            pred_unmasked_label = pred_unmasked.argmax(dim=1).item()

            processed_predictions_masked.append(pred_masked_label)
            processed_predictions_unmasked.append(pred_unmasked_label)
            
            # Save original and processed images
            original_category = cat[labels_val[i]]
            predicted_category = cat[pred_label]
            predicted_masked_category = cat[processed_predictions_masked[-1]]
            predicted_unmasked_category = cat[processed_predictions_unmasked[-1]]

            img_orl.save(os.path.join("vis_batch_adjust_with_ep/", f"{i}_orl_{original_category}_{predicted_category}.png"))
            img_masked.save(os.path.join("vis_batch_adjust_with_ep/", f"{i}_cpt{min_iou_cpt}_masked_{original_category}_{predicted_masked_category}.png"))
            img_unmasked.save(os.path.join("vis_batch_adjust_with_ep/", f"{i}_cpt{min_iou_cpt}_unmasked_{original_category}_{predicted_unmasked_category}.png"))
        else:
            # If correctly classified, add the same label to processed predictions
            processed_predictions_masked.append(pred_label)
            processed_predictions_unmasked.append(pred_label)

    # Calculate accuracies
    original_accuracy = evaluate_accuracy(original_predictions, labels_val) * 100
    processed_accuracy_masked = evaluate_accuracy(processed_predictions_masked, labels_val) * 100
    processed_accuracy_unmasked = evaluate_accuracy(processed_predictions_unmasked, labels_val) * 100

    # Output accuracies in percentage
    print(f"Original Accuracy: {original_accuracy:.4f}%")
    print(f"Processed Accuracy (Masked Area): {processed_accuracy_masked:.4f}%")
    print(f"Processed Accuracy (Unmasked Area): {processed_accuracy_unmasked:.4f}%")
    print("CPT Error Counts by Category:\n", cpt_error_counts)


if __name__ == '__main__':
    args = parser.parse_args()
    main()