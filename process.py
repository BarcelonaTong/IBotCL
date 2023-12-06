from loaders.get_loader import loader_generation
import torch
from configs import parser
from utils.tools import predict_hash_code, mean_average_precision
import os
from model.contrast.model_main import MainModel
import h5py


os.makedirs('data_map/', exist_ok=True)


def main():
    model = MainModel(args)
    device = torch.device(args.device)
    model.to(device)
    checkpoint = torch.load(os.path.join(args.output_dir,
            f"{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_" +
    f"{'use_slot_' + args.cpt_activation if not args.pre_train else 'no_slot'}.pt"), map_location="cuda:0")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    train_loader1, train_loader2, val_loader = loader_generation(args)

    # predict_hash_code -> all_output.numpy().astype("float32"), all_label.numpy().astype("float32"), round(accs/L, 4)
    print('Waiting for generating from database')
    database_hash, database_labels, database_preds, database_acc = predict_hash_code(args, model, train_loader2, device)
    print('Waiting for generating from test set')
    test_hash, test_labels, test_preds, test_acc = predict_hash_code(args, model, val_loader, device)
    
    # Create the HDF5 file to save the hashes and labels
    with h5py.File(f"data_map/{args.dataset}_{args.base_model}_cls{args.num_classes}_cpt{args.num_cpt}_{args.cpt_activation}.hdf5", "w") as f:
        f.create_dataset("database_hash", data=database_hash)
        f.create_dataset("database_labels", data=database_labels)
        # Save the predicted labels
        f.create_dataset("database_preds", data=database_preds) 
        f.create_dataset("test_hash", data=test_hash)
        f.create_dataset("test_labels", data=test_labels)
        # Save the predicted labels
        f.create_dataset("test_preds", data=test_preds)


if __name__ == '__main__':
    args = parser.parse_args()
    main()

