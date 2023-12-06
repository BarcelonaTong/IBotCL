import os
import h5py
import numpy as np
import pandas as pd
import torch
from configs import parser
from model.contrast.model_main import MainModel
from loaders.get_loader import load_all_imgs, get_transform

os.makedirs("batch_adjust_with_es_record/", exist_ok=True)


def load_expert_cpt_values(filepath):
    return pd.read_excel(filepath, index_col=0).values


def adjust_cpt_values(sample_cpt, expert_cpt, num_to_adjust):
    diff = np.abs(sample_cpt - expert_cpt)
    indices_to_adjust = np.argsort(-diff)[:num_to_adjust]
    max_diff_index = indices_to_adjust[0] if indices_to_adjust.size > 0 else None
    sample_cpt[indices_to_adjust] = expert_cpt[indices_to_adjust]
    return sample_cpt, max_diff_index


def main():
    # Load all images
    imgs_database, labels_database, imgs_val, labels_val, cat = load_all_imgs(args)
    
    expert_cpt_values = load_expert_cpt_values("vis_pp/es.xlsx")
    
    # Load the model and the validation data
    model = MainModel(args)
    device = torch.device('cuda:0')
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
                                     f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
                                     f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"),
                            map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    
    # Load the validation predictions and CPT values
    with h5py.File(f"data_map/{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", 'r') as f:
        val_hash = f['test_hash'][:]
        val_labels = f['test_labels'][:]
        val_preds = f['test_preds'][:]
        
    val_labels = val_labels.squeeze().astype(int)
    val_preds = val_preds.squeeze().astype(int)
    
    num_to_adjust = int(input("Enter the number of CPT values to adjust: "))
    
    columns = ["Original_Index"] + \
              [f"CPT_{i}_Before" for i in range(args.num_cpt)] + \
              [f"CPT_{i}_After" for i in range(args.num_cpt)] + \
              ["Category", "Prediction_Before", "Prediction_After"]

    adjustment_records = pd.DataFrame(columns=columns)
    
    # Initialize the error statistics
    cpt_error_counts = np.zeros((len(cat), args.num_cpt), dtype=int)
    
    # Adjust the CPT values for mispredicted samples
    mispredicted_indices = np.where(val_labels != val_preds)[0]
    for i in mispredicted_indices:
        sample_cpt_before = val_hash[i].copy()
        true_label = val_labels[i]
        predicted_label_before = val_preds[i]
        category_expert_cpt = expert_cpt_values[true_label]
        
        sample_cpt_after, max_diff_cpt_index = adjust_cpt_values(sample_cpt_before.copy(), category_expert_cpt, num_to_adjust)
        if max_diff_cpt_index is not None:
            cpt_error_counts[true_label][max_diff_cpt_index] += 1

        val_hash[i] = sample_cpt_after

        record = [i] + list(sample_cpt_before) + list(sample_cpt_after) + [true_label, predicted_label_before, None]
        adjustment_records.loc[len(adjustment_records)] = record

    # Recompute the predictions with the adjusted CPT values
    new_preds = []
    for i in range(len(val_labels)):
        sample_cpt = val_hash[i]
        with torch.no_grad():
            sample_cpt_tensor = torch.tensor(sample_cpt).float().unsqueeze(0).to(device)
            new_pred = model.cls(sample_cpt_tensor / 2 + 0.5)
            new_pp = torch.argmax(new_pred, dim=-1).item()
            new_preds.append(new_pp)
            
    for i, new_pred in enumerate(new_preds):
        if i in mispredicted_indices:
            original_index = adjustment_records[adjustment_records["Original_Index"] == i].index[0]
            adjustment_records.at[original_index, "Prediction_After"] = new_pred

    output_path = f"batch_adjust_with_es_record/val_{num_to_adjust}_cpt_adjust_changes.xlsx"
    adjustment_records.to_excel(output_path, index=False)

    # Calculate the new accuracy
    new_preds = np.array(new_preds)
    new_accuracy = np.mean(val_labels == new_preds)
    old_accuracy = np.mean(val_labels == val_preds)

    print(f"Old Accuracy: {old_accuracy:.4%}")
    print(f"New Accuracy: {new_accuracy:.4%}")
    print("CPT Error Counts by Category:\n", cpt_error_counts)


if __name__ == "__main__":
    args = parser.parse_args()
    main()
