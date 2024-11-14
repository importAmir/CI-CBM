import argparse
import json
import os
import numpy as np
from torch.utils.data import Subset
import utils
import data_utils
import cbm

# Initialize the argument parser
parser = argparse.ArgumentParser(description='Class-Incremental Concept Bottleneck Models')

# ** General settings **
parser.add_argument("--seed", type=int, default=1993, help="Random seed for reproducibility")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cuda', 'cpu')")

# ** Model settings **
parser.add_argument("--backbone", type=str, default="resnet18", help="Pretrained model to use as backbone") 
parser.add_argument("--strategy", type=str, default="backbone_prototype", 
                    choices=["naive", "full_rehearsal", "disjoint_pred", "bottleneck_prototype", "backbone_prototype"], 
                    help="CIL strategy")
parser.add_argument("--clip_name", type=str, default="ViT-L-16-SigLIP-384", help="CLIP model to use") 

# ** Data settings **
parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (e.g., 'cifar10', 'imagenet')")
parser.add_argument("--concept_set", type=str, default=None, help="Path to concept set file")

# ** Incremental Learning settings **
parser.add_argument("--n_experiences", type=int, default=5, help="Number of incremental experiences")
parser.add_argument("--half_split", action="store_true", help="Split classes: first half in the first experience, others across remaining ones")

# ** Directories **
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="Directory for saving activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="Directory for saving trained models")

# Parse arguments
args = parser.parse_args()

if args.half_split == True:
    args.n_experiences += 1

# Print the parsed arguments for review
print("Parsed arguments:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

if __name__=='__main__':
    # Set seed for reproducibility to ensure the results are consistent across runs
    utils.set_seed(args.seed)
    device = args.device
    
    # Load preprocess and validation dataset
    _, target_preprocess = data_utils.get_target_model(args.backbone, device)
    val_dataset_name = args.dataset + "_val"
    val_dataset = data_utils.get_data(val_dataset_name, preprocess=target_preprocess)
    
    # Load dataset class labels from file
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    
    # Split data into incremental learning groups
    grouped_classes, grouped_train_indices, grouped_test_indices, mapping_from_classes_to_cl_classes, mapping_from_cl_classes_to_classes = data_utils.split_data(
        n_experiences = args.n_experiences, 
        dataset_name = args.dataset, 
        classes = classes,
        half_split = args.half_split,
    )

    # Display the class mappings
    print("Mapping from original classes to grouped classes:", mapping_from_classes_to_cl_classes)
    print("Mapping from grouped classes to original classes:", mapping_from_cl_classes_to_classes)

    # Initialize a matrix to hold accuracy results
    result_matrix = np.zeros((args.n_experiences))
    
    # Iterate through each experience increment
    for exp_id in range(args.n_experiences):
        print("#"*20, f"experience {exp_id}", "#"*20)

        # Construct the directory path for the saved model
        load_dir = "{}/{}/{}_cbm/{}_backbone_{}_clip_name/{}_seed_{}_nexp/exp_{}".format(
            args.save_dir, args.strategy, args.dataset, args.backbone, args.clip_name, args.seed, args.n_experiences, exp_id
        )

        # Load training arguments to verify consistency
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            train_args = json.load(f)


        # Load the trained CBM model
        model = cbm.load_cbm(load_dir, device)

        # Accumulate test indices seen so far and evaluate accuracy on those indices
        seen_grouped_test_indices = sum([grouped_test_indices[i] for i in range(exp_id + 1)], []) 
        val_dataset_cl = Subset(val_dataset, seen_grouped_test_indices)
        accuracy = utils.get_accuracy_cbm(model, val_dataset_cl, device, mapping_from_cl_classes_to_classes)
        print(f"Accuracy of model in exp {exp_id}: {accuracy*100:.2f}%")
        result_matrix[exp_id] = accuracy
    

    print(result_matrix)

    # Calculate and display the average incremental accuracy
    avg_incremental_accuracy = np.mean(result_matrix)
    print(f"Average Incremental Accuracy: {avg_incremental_accuracy:.3f}")

    # Define the path for saving metrics
    metrics_file_path = os.path.join(
        args.save_dir,
        args.strategy,
        f"{args.dataset}_cbm",
        f"{args.backbone}_backbone_{args.clip_name}_clip_name",
        f"{args.seed}_seed_{args.n_experiences}_nexp",
        "metric.txt"
    )

    # Write each task's accuracy and average accuracy to file
    with open(metrics_file_path, 'w') as f:
        f.write(f"Average Accuracy (A_t) for tasks t = 0 to {len(result_matrix) - 1}:\n")
        for i, acc in enumerate(result_matrix):
            f.write(f"Task {i}: {acc:.3f}\n")
        
        f.write("\nAverage Incremental Accuracy (Ā_T):\n")
        f.write(f"{avg_incremental_accuracy:.3f}\n")
