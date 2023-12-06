from torchvision import datasets, transforms
from model.contrast.model_main import MainModel
from configs import parser
import torch
import os
from PIL import Image
from utils.tools import apply_colormap_on_image
from loaders.get_loader import load_all_imgs, get_transform
import shutil
from utils.tools import crop_center
import cv2

# Clear the folder if it exists and create it for fc_input tuning
shutil.rmtree("fc_input_tuning/", ignore_errors=True)
os.makedirs("fc_input_tuning/", exist_ok=True)


def print_cpt_values(cpt):
    print("Input Layer (CPT values):")
    for i, neuron_cpt in enumerate(cpt[0]):
        print(f"CPT_{i} ({neuron_cpt:.4f})")


def print_connection_weights(cpt, weights, labels):
    print("\nConnection Weights:")
    for j in range(len(weights[0])):
        for i in range(len(weights)):
            print(f"Weight from CPT_{i} to Class_{j} ({labels[j]}): {weights[i][j]:.2f}")
        print()


def main():
    # Load all images
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    print("All Categories:", cat)
    transform = get_transform(args)["val"]

    # Load model and weights
    model = MainModel(args, vis=True)
    device = torch.device('cuda:0')
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
                                         f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
                                         f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    index = args.index

    # Fetching the specific image and label
    data = imgs_database[index]
    label = labels_database[index]

    print('\n' + '=' * 60)
    print(f"{'True Label:':<20} {cat[label]}")
    print('=' * 60)

    # Image processing
    img_orl = Image.open(data).convert('RGB').resize([256, 256])
    img_orl2 = crop_center(img_orl, 224, 224)
    img_orl2.save("fc_input_tuning/origin.png")
    cpt, pred, att, update, last_layer_weights = model(transform(img_orl).unsqueeze(0).to(device), None, None,
                                                        return_cpt=True, loc="fc_input_tuning")
    pp = torch.argmax(pred, dim=-1)

    print('=' * 60)
    print(f"{'Predicted Class:':<20} {cat[pp]}")
    print('=' * 60 + '\n')

    # Apply the heatmap
    for id in range(args.num_cpt):
        slot_image = cv2.imread(f"fc_input_tuning/0_slot_{id}.png", cv2.IMREAD_GRAYSCALE)
        heatmap_only, heatmap_on_image = apply_colormap_on_image(img_orl2, slot_image, 'jet')
        heatmap_on_image.save(f"fc_input_tuning/0_slot_mask_{id}.png")

    # Initialize random weights for demonstration
    weights = torch.randn((5, 4))

    print("Original Neural Network Structure:\n")
    print_cpt_values(cpt)
    print_connection_weights(cpt, weights, cat)
    print('\n' + '=' * 60 + '\n')

    adjust = True
    while adjust:
        user_input = input("Adjust cpt value by entering 'cpt_index new_value' or type 'exit' to quit: ")
        if user_input.lower() == "exit":
            adjust = False
            continue
        try:
            parts = user_input.split()
            if len(parts) != 2:
                raise ValueError("Invalid input format. Please enter two values: cpt_index and new_value.")
            cpt_index, new_value = int(parts[0]), float(parts[1])
            if cpt_index < 0 or cpt_index >= len(cpt[0]):
                raise ValueError(f"CPT index must be between 0 and {len(cpt[0]) - 1}.")

            # Update cpt value
            cpt[0][cpt_index] = new_value
            # Compute new prediction
            with torch.no_grad():
                new_pred = model.cls(cpt / 2 + 0.5) # !!!
                new_pp = torch.argmax(new_pred, dim=-1)
            print("Updated CPT values:")
            print_cpt_values(cpt)
            print("New predicted class:", cat[new_pp])
            print('\n' + '=' * 60 + '\n')
        except ValueError as e:
            print(e)


if __name__ == '__main__':
    args = parser.parse_args()
    main()