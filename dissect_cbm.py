import argparse
import utils
import torch 
import os 
import pandas as pd
from sentence_transformers import SentenceTransformer
import clip 
import numpy as np

import similarity
import data_utils

parser = argparse.ArgumentParser(description='Label-Free Concept Bottleneck Model for Class Incremental Learning')
parser.add_argument("--clip_name", type=str, default="ViT-B/16", choices=utils.CLIP_MODEL_NAMES, help="CLIP or SigLIP model to use")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
parser.add_argument("--seed", type=int, default=1993, help="Random seed for reproducibility")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Pretrained model to use as backbone")
parser.add_argument("--strategy", type=str, default="naive", help="Our CL strategy")
parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name")
parser.add_argument("--n_experiences", type=int, default=5, help="Number of incremental experiences")
parser.add_argument("--save_dir", type=str, default='saved_models', help="Directory for saving trained models")
parser.add_argument("--concept_set", type=str, default=None, help="Path to concept set file")
parser.add_argument("--feature_layer", type=str, default='layer4', help="Layer to collect activations from (second to last layer name)")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="Directory for saving activations")
parser.add_argument("--conceptnet_flag", action="store_true")
parser.add_argument("--half_split", action="store_true", help="Split classes: first half in the first experience, others across remaining ones")

args = parser.parse_args()

original_n_experiences = args.n_experiences
if args.half_split:
    args.n_experiences += 1

path_n_experiences = original_n_experiences + 1 if args.half_split else args.n_experiences

if args.concept_set == None:
    args.concept_set = "data/concept_sets/{}_filtered.txt".format(args.dataset)

print("Parsed arguments:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

if __name__=='__main__':
    utils.set_seed(args.seed)
    device = args.device

    clip_model, _ = clip.load("ViT-B/16", device=device)
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    
    # Define similarity function
    similarity_fn = similarity.soft_wpmi
    # similarity_fn = similarity.cos_similarity_cubed

    # Define dataset names for training and validation
    d_val = args.dataset + "_val"

    # Load dataset classes and concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    concept_eval_output = {
        "unit": list(range(len(concepts))),
        "GT concepts" : [],
    }
    # Get save names for features
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(
        clip_name = args.clip_name, 
        target_name = args.backbone,
        target_layer = args.feature_layer, 
        d_probe = d_val, 
        concept_set = args.concept_set, 
        pool_mode = "avg",
        save_dir = args.activation_dir
    )

    # Split dataset into groups for incremental learning
    grouped_classes, _, _, _, _ = data_utils.split_data(
        n_experiences=args.n_experiences, 
        dataset_name=args.dataset, 
        classes=classes
    )
    print(grouped_classes)

    # Initialize lists for concepts  across experiences
    concepts_cl = []
    concepts_cl_indx = {}

    # Load features for validation
    with torch.no_grad():
        val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
        val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
        val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True)
        text_features = torch.load(text_save_name, map_location="cpu").float()
        text_features /= torch.norm(text_features, dim=1, keepdim=True)
        val_clip_features = val_image_features @ text_features.T

        del val_image_features, text_features
    
    # Iterate over incremental experiences
    for exp_id in range(args.n_experiences):
        print("#"*20, f"experience {exp_id}", "#"*20)

        # Get concepts for selected classes in the current experience
        selected_concepts = data_utils.get_concepts_for_classes(grouped_classes[exp_id], args.dataset, args.conceptnet_flag)

        # Merge concepts while tracking duplicates
        concepts_cl, new_repeated = data_utils.merge_concepts(concepts_cl, selected_concepts)

        # Find indices of merged concepts in the global concept set
        concepts_cl_indx[exp_id] = data_utils.find_indices_of_concepts(selected_concepts, concepts_cl)



        # Use path_n_experiences for consistency with saved model paths
        load_dir = "{}/{}/{}_cbm/{}_backbone_{}_clip_name/{}_seed_{}_nexp/exp_{}".format(args.save_dir, args.strategy, args.dataset, args.backbone, args.clip_name, args.seed, path_n_experiences, exp_id)
        
        # Load W_c first to get its actual size (it has the full accumulated size up to this experience)
        W_c = torch.load(os.path.join(load_dir ,"W_c.pt"), map_location="cpu")
        
        # Load the saved concepts.txt to get the exact concept list that was used when saving W_c
        # This ensures we use the same concepts that were used during training
        concepts_file = os.path.join(load_dir, "concepts.txt")
        if os.path.exists(concepts_file):
            with open(concepts_file, "r") as f:
                saved_concepts_cl = [line.strip() for line in f.readlines()]
            # Replace concepts_cl with saved concepts to ensure exact match
            if len(saved_concepts_cl) != len(concepts_cl) or saved_concepts_cl != concepts_cl:
                print(f"Info: Using saved concepts from file (length: {len(saved_concepts_cl)}) instead of incrementally built concepts (length: {len(concepts_cl)}).")
                concepts_cl = saved_concepts_cl
                # Recompute concepts_cl_indx for all experiences up to current one
                for prev_exp in range(exp_id + 1):
                    prev_selected_concepts = data_utils.get_concepts_for_classes(grouped_classes[prev_exp], args.dataset, args.conceptnet_flag)
                    concepts_cl_indx[prev_exp] = data_utils.find_indices_of_concepts(prev_selected_concepts, concepts_cl)
        
        # Build projection layer with the size of the loaded W_c
        # This ensures the layer size matches the saved weights exactly
        proj_layer = torch.nn.Linear(
                in_features = val_target_features.shape[1], 
                out_features = W_c.shape[0],  # Use the size from loaded W_c
                bias = False
            ).to(args.device)
        W_c = W_c.to(args.device)
        # print("W_c.shape", W_c.shape)
        # print("W_c norm", torch.norm(W_c,dim=1))
        proj_layer.load_state_dict({"weight":W_c})

        # Project validation target features to concept space
        # val_target_concepts shape: (num_val_samples, num_concept_units)
        # Each column represents activations of one concept unit across all validation samples
        val_target_concepts = proj_layer(val_target_features.to(args.device).detach())

        # Compute similarities between concept unit activations and CLIP concept-image alignments
        # soft_wpmi computes: for each concept unit, find top-k samples where it's most active,
        # then compute mutual information with CLIP features for those samples
        # similarities shape: (num_concept_units, num_concepts)
        similarities = similarity_fn(val_clip_features.to("cpu"), val_target_concepts.to("cpu"))
        
        # Ensure similarities is (num_concept_units, num_concepts)
        # If soft_wpmi returns (num_concepts, num_concept_units), transpose it
        if similarities.shape[0] != len(concepts_cl):
            # If first dimension doesn't match number of units, likely need to transpose
            if similarities.shape[1] == len(concepts_cl):
                similarities = similarities.T
            else:
                raise ValueError(f"Unexpected similarities shape: {similarities.shape}, expected ({len(concepts_cl)}, {len(concepts)})")

        # For each concept unit, find the most similar concept (top-1)
        # similarities[i, j] = similarity between unit i and concept j
        # Find max along concept dimension (dim=1) to get best concept for each unit
        vals, concept_ids = torch.max(similarities, dim=1)
        
        # Get top-k (k=5 and k=10) concept indices for each unit
        # top5_ids and top10_ids shape: (num_concept_units, k) - these are concept indices
        _, top5_ids = torch.topk(similarities, k=min(5, similarities.size(1)), dim=1)
        _, top10_ids = torch.topk(similarities, k=min(10, similarities.size(1)), dim=1)
        
        # For top-k accuracy: for each concept unit i, check if its ground truth concept
        # appears in the top-k most similar concepts
        top5_correct = torch.zeros(len(concepts_cl), dtype=torch.bool)
        top10_correct = torch.zeros(len(concepts_cl), dtype=torch.bool)
        
        # Find ground truth concept indices in the full concept list
        # concepts_cl[i] is the ground truth concept for unit i
        # We need to find the index of concepts_cl[i] in the full concepts list
        for i in range(len(concepts_cl)):
            gt_concept = concepts_cl[i]
            try:
                gt_concept_idx = concepts.index(gt_concept)
            except ValueError:
                # If ground truth concept not in full concept list, skip
                top5_correct[i] = False
                top10_correct[i] = False
                continue
            
            # Check if ground truth concept is in top-5 for this unit
            unit_in_top5 = (top5_ids[i] == gt_concept_idx).any()
            # Check if ground truth concept is in top-10 for this unit
            unit_in_top10 = (top10_ids[i] == gt_concept_idx).any()
            
            top5_correct[i] = unit_in_top5.item()
            top10_correct[i] = unit_in_top10.item()
        
        top5_acc = top5_correct.float().mean().item()
        top10_acc = top10_correct.float().mean().item()

        del similarities
        torch.cuda.empty_cache()
        
        # Get selected concepts based on CLIP-dissect: for each unit, get its most similar concept
        descriptions = [concepts[int(idx)] for idx in concept_ids]

        concept_eval_output[f"Experience {exp_id}"] = descriptions

        concept_eval_output[f"Experience {exp_id}"] = concept_eval_output[f"Experience {exp_id}"] + (['N/A'] * (len(concepts) - len(concept_eval_output[f"Experience {exp_id}"])))
        # Save the unit number, the concept assigned by CLIP-dissect, the similarity score of this concept, and the ground truth concept.
        # Each row corresponds to one concept unit
        outputs = {
            "unit": list(range(len(concepts_cl))),
            "description": descriptions,
            "similarity": vals.cpu().numpy(),
            "gt": concepts_cl
        }

        df = pd.DataFrame(outputs)
        # Use path_n_experiences for consistency with saved model paths
        save_path = os.path.join(args.save_dir, args.strategy, f"{args.dataset}_cbm", f"{args.backbone}_backbone_{args.clip_name}_clip_name", f"{args.seed}_seed_{path_n_experiences}_nexp", f'exp_{exp_id}')
        df.to_csv(os.path.join(save_path,"descriptions.csv"), index=False)

        # Calculate additional metrics like cosine similarity and accuracy
        clip_cos, mpnet_cos = utils.get_cos_similarity(
            preds = descriptions, 
            gt = concepts_cl, 
            clip_model = clip_model, 
            mpnet_model = mpnet_model, 
            device = args.device, 
            batch_size = 200,
            reduce = "none"
        )
        correct_pred = np.array([1 if descriptions[i] == concepts_cl[i] else 0 for i in range(len(descriptions))])
        print(f"clip_cos: {float(torch.mean(clip_cos)):.3f}, mpnet_cos: {float(np.mean(mpnet_cos)):.3f}, accuracy: {np.mean(correct_pred):.3f}, top5_acc: {top5_acc:.3f}, top10_acc: {top10_acc:.3f}")

        # Write results to dissect_results.txt
        # Use the same directory structure as saved models
        # Use path_n_experiences for consistency with saved model paths
        dissect_results_file_path = os.path.join(args.save_dir, args.strategy, f"{args.dataset}_cbm", f"{args.backbone}_backbone_{args.clip_name}_clip_name", f"{args.seed}_seed_{path_n_experiences}_nexp", "dissect_results.txt")
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(dissect_results_file_path), exist_ok=True)
        if exp_id == 0:
            with open(dissect_results_file_path, "w") as file:
                file.write(f"Experience {exp_id}:\n")
                file.write(f"clip_cos: {torch.mean(clip_cos):.3f}, mpnet_cos: {np.mean(mpnet_cos):.3f}, accuracy: {np.mean(correct_pred):.3f}, top5_acc: {top5_acc:.3f}, top10_acc: {top10_acc:.3f}\n\n")
        else:
            with open(dissect_results_file_path, "a") as file:
                file.write(f"Experience {exp_id}:\n")
                file.write(f"clip_cos: {torch.mean(clip_cos):.3f}, mpnet_cos: {np.mean(mpnet_cos):.3f}, accuracy: {np.mean(correct_pred):.3f}, top5_acc: {top5_acc:.3f}, top10_acc: {top10_acc:.3f}\n")
                for i in range(exp_id + 1):
                    file.write(f"similarity for concepts related to task {i}:\n")
                    # Compute top-k accuracy for concepts in this task
                    task_top5_correct = top5_correct[concepts_cl_indx[i]]
                    task_top10_correct = top10_correct[concepts_cl_indx[i]]
                    task_top5_acc = task_top5_correct.float().mean().item() if len(task_top5_correct) > 0 else 0.0
                    task_top10_acc = task_top10_correct.float().mean().item() if len(task_top10_correct) > 0 else 0.0
                    file.write(f"clip_cos: {torch.mean(clip_cos[concepts_cl_indx[i]]):.3f}, mpnet_cos: {np.mean(mpnet_cos[concepts_cl_indx[i]]):.3f}, accuracy: {np.mean(correct_pred[concepts_cl_indx[i]]):.3f}, top5_acc: {task_top5_acc:.3f}, top10_acc: {task_top10_acc:.3f}\n")
                file.write("\n")

    concept_eval_output["GT concepts"] = concepts_cl
    # Use path_n_experiences for consistency with saved model paths
    concept_evaluation_file_path = os.path.join(args.save_dir, args.strategy, f"{args.dataset}_cbm", f"{args.backbone}_backbone_{args.clip_name}_clip_name", f"{args.seed}_seed_{path_n_experiences}_nexp", "concept_evaluation.txt")
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(concept_evaluation_file_path), exist_ok=True)
    df = pd.DataFrame(concept_eval_output)
    df.to_csv(concept_evaluation_file_path, index=False)