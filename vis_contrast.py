import os
import shutil
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from PIL import Image
import cv2
import h5py

from model.contrast.model_main import MainModel
from configs import parser
from loaders.get_loader import load_all_imgs, get_transform
from utils.tools import apply_colormap_on_image, for_retrival, attention_estimation, crop_center
from utils.draw_tools import draw_bar, draw_plot
from tqdm import tqdm
from batch_adjust_with_ep import calculate_iou

shutil.rmtree("vis/", ignore_errors=True)
shutil.rmtree("vis_pp/", ignore_errors=True)
os.makedirs("vis/", exist_ok=True)
os.makedirs("vis_pp/", exist_ok=True)

np.set_printoptions(suppress=True)


def main():
    # Load all images
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    transform = get_transform(args)["val"]

    # Load model and weights
    model = MainModel(args, vis=True)
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

    index = args.index

    # # Attention statistic (commented out)
    # name = "Yellow_headed_Blackbird"
    # att_record = attention_estimation(imgs_database, labels_database, model, transform, device, name=name)
    # draw_plot(att_record, name)

    data = imgs_database[index]
    label = labels_database[index]

    # print('\n' + '=' * 60)
    # print(f"{'True Label:':<20} {cat[label]}")
    # print('=' * 60)

    # img_orl = Image.open(data).convert('RGB').resize([299, 299])
    # # img_orl2 = crop_center(img_orl, 224, 224)
    # img_orl2 = img_orl
    # img_orl2.save(f"vis/origin.png")
    # cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None, mask_size=299)
    # pp = torch.argmax(pred, dim=-1)

#     print('=' * 60)
#     print(f"{'Predicted Class:':<20} {cat[pp]}")
#     print('=' * 60 + '\n')

#     w = model.state_dict()["cls.weight"][label]
#     w_numpy = np.around(torch.tanh(w).cpu().detach().numpy(), 4)
#     ccc = np.around(cpt.cpu().detach().numpy(), 4)

#     print('\n' + '=' * 60)
#     print("Weight:", w_numpy)
#     print("Cpt:", ccc)
#     print("Sum:", (ccc / 2 + 0.5) * w_numpy)

    # # Draw attention maps for each concept
    # for id in range(args.num_cpt):
    #     slot_image_path = f"vis/0_slot_{id}.png"
    #     slot_image = cv2.imread(slot_image_path, cv2.IMREAD_GRAYSCALE)
    #     heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
    #     heatmap_on_image.save(f"vis/0_slot_mask_{id}.png")

    # Load predicted labels as well
    with h5py.File(f"data_map/{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", 'r') as f1:
        database_hash = f1["database_hash"][:]
        database_labels = f1["database_labels"][:]
        # Load predicted labels
        database_preds = f1["database_preds"][:]

    for j in tqdm(range(args.num_cpt), desc=f"Generating concept samples"):
        for c, category_name in enumerate(cat):
            root = os.path.join("vis_pp", f"cpt{j}", category_name)
            os.makedirs(root, exist_ok=True)

            # Select indices for all samples of current category c that were predicted correctly
            correct_preds_indices = np.where((database_labels[:, 0] == c) & (database_preds[:, 0] == c))[0]
            # Sort the responses for current cpt and category c to get top samples
            selected = database_hash[correct_preds_indices, j]
            ids = np.argsort(-selected, axis=0)
            top_indices = ids[:args.top_samples]
            top_idx_for_category = correct_preds_indices[top_indices]

            # Save visualizations of the top samples for the current cpt and category
            for i, idx in enumerate(top_idx_for_category):
                img_path = imgs_database[idx]
                img_orl = Image.open(img_path).convert('RGB')
                # img_orl2 = crop_center(img_orl, 224, 224)
                cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category_name, j], mask_size=299)
                img_orl.save(os.path.join(root, f"orl_{i}_{category_name}.png"))
                slot_image = np.array(Image.open(os.path.join(root, f"mask_{i}_{category_name}.png")), dtype=np.uint8)
                heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl, slot_image, 'jet')
                heatmap_on_image.save(os.path.join(root, f"jet_{i}_{category_name}.png"))
                
    # ********************************* EP *********************************
    # Calculate the weighted average mask for each category for each cpt
    for j in tqdm(range(args.num_cpt), desc=f"Calculating the weighted average mask"):
        for c, category_name in enumerate(cat):
            root = os.path.join("vis_pp", f"cpt{j}", category_name)

            mask_files = [f for f in os.listdir(root) if f.startswith('mask_') and f.endswith('.png')]
            if not mask_files:
                print(f"No mask files found for category {category_name} in {root}, skipping.")
                continue

            # Compute weights for all masks of the current category
            weights = []
            for mask_file in mask_files:
                mask_path = os.path.join(root, mask_file)
                mask_img = np.array(Image.open(mask_path), dtype=float)
                weight = np.sum(mask_img) / 255.0
                weights.append(weight)
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Calculate the weighted average mask for the current category
            weighted_avg_mask = None
            for i, mask_file in enumerate(mask_files):
                mask_path = os.path.join(root, mask_file)
                mask_img = np.array(Image.open(mask_path), dtype=float)
                if weighted_avg_mask is None:
                    weighted_avg_mask = mask_img * weights[i]
                else:
                    weighted_avg_mask += mask_img * weights[i]
            weighted_avg_mask = np.round(weighted_avg_mask).astype(np.uint8)

            # Save the weighted average mask image for the current category
            weighted_avg_mask_img = Image.fromarray(weighted_avg_mask)
            expert_mask_path = os.path.join(root, f"weighted_expert_mask_{category_name}.png")
            weighted_avg_mask_img.save(expert_mask_path)

            # # Apply the weighted average mask to all corresponding original images
            # for orl_file in os.listdir(root):
            #     if orl_file.startswith('orl_') and orl_file.endswith('.png'):
            #         orl_path = os.path.join(root, orl_file)
            #         orl_img = Image.open(orl_path)
            #         heatmap_only, heatmap_on_image = apply_colormap_on_image(orl_img, weighted_avg_mask, 'jet')
            #         output_path = os.path.join(root, f"weighted_expert_{orl_file}")
            #         heatmap_on_image.save(output_path)
            
    # ep_stats   
    iou_stats = []

    # Repeat the process of calculating the weighted average mask for each cpt and category
    for j in tqdm(range(args.num_cpt), desc=f"Calculating ep stats"):
        for c, category_name in enumerate(cat):
            root = os.path.join("vis_pp", f"cpt{j}", category_name)
            mask_files = [f for f in os.listdir(root) if f.startswith('mask_') and f.endswith('.png')]

            if not mask_files:
                print(f"No mask files found for category {category_name} in {root}, skipping.")
                continue

            # Read the weighted average mask
            weighted_avg_mask_path = os.path.join(root, f"weighted_expert_mask_{category_name}.png")
            weighted_avg_mask = np.array(Image.open(weighted_avg_mask_path), dtype=bool)

            # Calculate IOU for each mask with the weighted average mask
            iou_scores = []
            for mask_file in mask_files:
                mask_path = os.path.join(root, mask_file)
                mask_img = np.array(Image.open(mask_path), dtype=bool)
                iou_score = calculate_iou(mask_img, weighted_avg_mask)
                iou_scores.append(iou_score)

            # Calculate mean and standard deviation
            iou_mean = np.mean(iou_scores)
            iou_std = np.std(iou_scores)

            # Save the results
            iou_stats.append({
                'cpt': j,
                'category': category_name,
                'iou_mean': iou_mean,
                'iou_std': iou_std
            })

    # Save the results as an Excel file
    iou_df = pd.DataFrame(iou_stats)
    iou_df.to_excel(os.path.join("vis_pp", "ep_stats.xlsx"), index=False)
    
    # ********************************* ES *********************************
    # Initialize a structure to hold the expert values for each category and each cpt
    expert_values = np.zeros((len(cat), args.num_cpt))

    # Calculate the expert values for each category and each cpt
    for c, category_name in enumerate(cat):
        # Select indices for all samples of the current category c that were predicted correctly
        correct_preds_indices = np.where((database_labels[:, 0] == c) & (database_preds[:, 0] == c))[0]

        # For each cpt, compute the average value of the cpt for the samples that were correctly predicted for the current category
        for j in range(args.num_cpt):
            expert_values[c, j] = np.mean(database_hash[correct_preds_indices, j])

    # Convert the expert values into a DataFrame for writing to Excel
    expert_values_df = pd.DataFrame(expert_values, index=[f"Category {i}" for i in range(len(cat))], columns=[f"CPT {j}" for j in range(args.num_cpt)])

    # Write the DataFrame to an Excel file
    expert_values_df.to_excel("vis_pp/es.xlsx", index=True)
    
    # Initialize a structure to hold the mean and standard deviation for each category and each cpt
    mean_diff = np.zeros((len(cat), args.num_cpt))
    std_diff = np.zeros((len(cat), args.num_cpt))
    
    # es stats
    es_stats = []

    # Calculate the standard deviation for each category and each cpt
    for c, category_name in enumerate(cat):
        # Select indices for all samples of the current category c
        category_indices = np.where((database_labels[:, 0] == c) & (database_preds[:, 0] == c))[0]

        # For each cpt, compute the standard deviation
        for j in range(args.num_cpt):
            cpt_values = database_hash[category_indices, j]
            std_diff = np.std(cpt_values)

            # Append the results to the list
            es_stats.append({
                'cpt': j,
                'category': category_name,
                'std': std_diff
            })

    # Convert the list into a DataFrame for writing to Excel
    es_stats_df = pd.DataFrame(es_stats)

    # Write the DataFrame to an Excel file
    es_stats_df.to_excel("vis_pp/es_stats.xlsx", index=False)


if __name__ == '__main__':
    args = parser.parse_args()
    main()