import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from configs import parser
from model.contrast.model_main import MainModel
from loaders.get_loader import load_all_imgs, get_transform


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


def apply_masks_to_image(image, masks, threshold):
    for mask in masks:
        mask_array = np.array(mask)
        mask_array = mask_array > threshold
        image = np.where(mask_array[..., None], 0, image)
    return image


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

    # New functionality implementation
    original_preds = []
    masked_preds = []
    
    # Threshold for mask processing
    mask_threshold = int(input("Enter the mask intensity threshold value (0-255): "))

    # 1. Processing original validation set samples
    for i, img_path in enumerate(tqdm(imgs_val, desc="Processing original images")):
        img_orl = Image.open(img_path).convert('RGB')
        input_tensor = transform(img_orl).unsqueeze(0).to(device)
        cpt, pred, att, update = model(input_tensor)
        original_preds.append(pred[0][labels_val[i]].item())

    # 2. Reading masks and applying them to images
    for i, (img_path, sample_masks) in enumerate(zip(tqdm(imgs_val, desc="Applying masks"), masks)):
        img_orl = Image.open(img_path).convert('RGB')
        img_array = np.array(img_orl)
        masked_image = apply_masks_to_image(img_array, sample_masks, mask_threshold)
        masked_image_pil = Image.fromarray(masked_image)
        input_tensor = transform(masked_image_pil).unsqueeze(0).to(device)
        cpt, pred, att, update = model(input_tensor)
        masked_preds.append(pred[0][labels_val[i]].item())

    # 4. Calculating the average difference in predictions before and after processing
    pred_diffs = [abs(o - m) for o, m in zip(original_preds, masked_preds)]
    mean_diff = np.mean(pred_diffs)
    print(f"Average difference in predictions: {mean_diff}")


if __name__ == "__main__":
    args = parser.parse_args()
    main()
