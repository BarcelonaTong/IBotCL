import os
import shutil
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from configs import parser
from model.contrast.model_main import MainModel
from loaders.get_loader import load_all_imgs, get_transform

shutil.rmtree("vis_val_masks/", ignore_errors=True)
os.makedirs("vis_val_masks/", exist_ok=True)


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

    # Iterate over all validation images
    for i, img_path in enumerate(tqdm(imgs_val, desc="Generating masks for validation images")):
        # Get the category label for the current image
        label_index = labels_val[i]
        category_name = cat[label_index]
        
        for j in range(args.num_cpt):
            root = os.path.join("vis_val_masks/", f"cpt{j}", category_name)
            os.makedirs(root, exist_ok=True)
            
            # Load and preprocess image
            img_orl = Image.open(img_path).convert('RGB')
            # Process image with the model
            cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, [i, category_name, j], loc2="vis_val_masks/", mask_size=299)
            

if __name__ == '__main__':
    args = parser.parse_args()
    main()