import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from configs import parser
from model.contrast.model_main import MainModel
from loaders.get_loader import load_all_imgs, get_transform


def generate_confusion_matrix(true_labels, pred_labels, num_classes):
    """
    Generate a confusion matrix.

    :param true_labels: List of true labels.
    :param pred_labels: List of predicted labels.
    :param num_classes: Number of classes.
    :return: Confusion matrix as a 2D NumPy array.
    """
    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(true_labels, pred_labels):
        conf_matrix[true][pred] += 1
    return conf_matrix


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
    
    true_labels = []
    pred_labels = []
    
    for i, (img_path, true_label) in enumerate(tqdm(zip(imgs_val, labels_val), desc="Processing original images")):
        img_orl = Image.open(img_path).convert('RGB')
        input_tensor = transform(img_orl).unsqueeze(0).to(device)
        cpt, pred, att, update = model(input_tensor)
        pred_label = torch.argmax(pred, dim=1).item()
        true_labels.append(true_label)
        pred_labels.append(pred_label)
        
    conf_matrix = generate_confusion_matrix(true_labels, pred_labels, len(cat))
    print("\nConfusion matrix:\n", conf_matrix)
    
    error_percentages = {}
    for i, category in enumerate(cat):
        total_samples = sum(conf_matrix[i])
        correct_predictions = conf_matrix[i][i]
        incorrect_predictions = total_samples - correct_predictions
        error_percentage = (incorrect_predictions / total_samples) * 100
        error_percentages[category] = error_percentage

    print("\nError Percentages by Category:")
    for category, percentage in error_percentages.items():
        formatted_percentage = f"{percentage:.4f}%"
        print(f"{category}: {formatted_percentage}")
    
    
if __name__ == "__main__":
    args = parser.parse_args()
    main()
