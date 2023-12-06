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
os.makedirs("vis/", exist_ok=True)

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

    data = imgs_val[index]
    label = labels_val[index]

    # print('\n' + '=' * 60)
    # print(f"{'True Label:':<20} {cat[label]}")
    # print('=' * 60)

    img_orl = Image.open(data).convert('RGB').resize([299, 299])
    # img_orl2 = crop_center(img_orl, 224, 224)
    img_orl2 = img_orl
    img_orl2.save(f"vis/origin.png")
    cpt, pred, att, update = model(transform(img_orl).unsqueeze(0).to(device), None, None, mask_size=299)
    pp = torch.argmax(pred, dim=-1)

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

    # Draw attention maps for each concept
    for id in range(args.num_cpt):
        slot_image_path = f"vis/0_slot_{id}.png"
        slot_image = cv2.imread(slot_image_path, cv2.IMREAD_GRAYSCALE)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
        heatmap_on_image.save(f"vis/0_slot_mask_{id}.png")
        

if __name__ == '__main__':
    args = parser.parse_args()
    main()