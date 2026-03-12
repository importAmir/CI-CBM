import os
import argparse
import torch 
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset

import utils
import similarity
import data_utils
import training_utils
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga

parser = argparse.ArgumentParser(description='Class-Incremental Concept Bottleneck Models')
parser.add_argument("--seed", type=int, default=1993, help="Random seed for reproducibility")
parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation (e.g., 'cuda', 'cpu')")
parser.add_argument("--backbone", type=str, default="resnet18", help="Pretrained model to use as backbone") 
parser.add_argument("--strategy", type=str, default="backbone_prototype", 
                    choices=["naive", "full_rehearsal", "disjoint_pred", "bottleneck_prototype", "backbone_prototype"], 
                    help="CIL strategy")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", choices=utils.CLIP_MODEL_NAMES, help="CLIP or SigLIP model to use") 
parser.add_argument("--feature_layer", type=str, default='layer4', help="Layer for activations")
parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (e.g., 'cifar10', 'imagenet')")
parser.add_argument("--concept_set", type=str, default=None, help="Path to concept set file")
parser.add_argument("--conceptnet_flag", action="store_true")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size for saving activations")
parser.add_argument("--proj_steps", type=int, default=1000, help="Projection layer training steps")
parser.add_argument("--proj_eval_freq", type=int, default=50, help="Frequency for evaluating projection layer")
parser.add_argument("--proj_batch_size", type=int, default=50000, help="Batch size for learning projection layer")
parser.add_argument("--distill_weight", type=float, default=2.0,
                    help="Weight applied to the LwF distillation loss term when training the projection layer (only used for LwF runs).")
parser.add_argument("--saga_batch_size", type=int, default=256, help="Batch size for fitting final layer")
parser.add_argument("--SAGA_lr", type=float, default=0.1, help="Learning rate for prediction layer (SAGA algorithm)")
parser.add_argument("--n_iters", type=int, default=1000, help="Iterations for final layer solver")
parser.add_argument("--lam", type=float, default=0.0007, help="Sparsity regularization parameter")
parser.add_argument("--n_experiences", type=int, default=5, help="Number of incremental experiences")
parser.add_argument("--half_split", action="store_true", help="Split classes: first half in the first experience, others across remaining ones")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="Directory for saving activations")
parser.add_argument("--save_dir", type=str, default='saved_models', help="Directory for saving trained models")

args = parser.parse_args()

if not args.concept_set:
    if  args.conceptnet_flag == True:
        print('using concetps from conceptnet')
        args.concept_set = f"data/concept_sets/conceptnet/{args.dataset}_filtered.txt"
    else:
        args.concept_set = f"data/concept_sets/{args.dataset}_filtered.txt"

if args.half_split:
    args.n_experiences += 1

print("Parsed arguments:")
for key, value in vars(args).items():
    print(f"{key}: {value}")

if __name__=='__main__':
    utils.set_seed(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    # Define the similarity function (cosine similarity cubed in this case)
    similarity_fn = similarity.cos_similarity_cubed_single

    # Define dataset names for training and validation
    d_train = args.dataset + "_train"
    d_val = args.dataset + "_val"

    # Load dataset classes and concept set
    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")
    with open(args.concept_set) as f:
        concepts = f.read().split("\n")

    # Save activations for both training and validation datasets
    for d_probe in [d_train, d_val]:
        utils.save_activations(
            clip_name = args.clip_name, 
            target_name = args.backbone, 
            target_layers = [args.feature_layer], 
            d_probe = d_probe,
            concept_set = args.concept_set, 
            batch_size = args.batch_size, 
            device = args.device, 
            pool_mode = "avg", 
            save_dir = args.activation_dir,
        )
    
    # Get file paths for saved activations for both train and validation
    target_save_name, clip_save_name, text_save_name = utils.get_save_names(
        clip_name = args.clip_name, 
        target_name = args.backbone, 
        target_layer = args.feature_layer,
        d_probe = d_train, 
        concept_set = args.concept_set, 
        pool_mode = "avg", 
        save_dir = args.activation_dir
    )
    val_target_save_name, val_clip_save_name, text_save_name =  utils.get_save_names(
        clip_name = args.clip_name, 
        target_name = args.backbone,
        target_layer = args.feature_layer, 
        d_probe = d_val, 
        concept_set = args.concept_set, 
        pool_mode = "avg",
        save_dir = args.activation_dir
    )
    
    if args.backbone.startswith('SelfPromptDeit_mytiny'):
        grouped_classes, grouped_train_indices, grouped_test_indices, mapping_from_classes_to_cl_classes, _ = data_utils.split_data_SelfPromptDeit(
            n_experiences = args.n_experiences, 
            dataset_name = args.dataset, 
            classes = classes,
            half_split = args.half_split,
        )
    else:
        # Split the dataset into groups based on the incremental learning setup
        grouped_classes, grouped_train_indices, grouped_test_indices, mapping_from_classes_to_cl_classes, _ = data_utils.split_data(
            n_experiences = args.n_experiences, 
            dataset_name = args.dataset, 
            classes = classes,
            half_split = args.half_split,
        )
    
    # Print mapping from original classes to grouped classes
    print("Mapping from original classes to grouped classes:", mapping_from_classes_to_cl_classes)
    
    # Initialize lists for concepts and classes across experiences
    concepts_cl = []
    classes_cl = []

    # Iterate over the incremental learning experiences
    for exp_id in range(args.n_experiences):
        print("#"*20, f"experience {exp_id}", "#"*20) # Print experience number for tracking

        # Accumulate classes for the current experience (incrementally adding classes)
        classes_cl = classes_cl + grouped_classes[exp_id]

        # Get the concepts corresponding to the classes for the current experience
        selected_concepts = data_utils.get_concepts_for_classes(grouped_classes[exp_id], args.dataset, args.conceptnet_flag)
        # print(f"Selected concepts for experience {exp_id}: {selected_concepts}")

        # Merge new concepts with the already accumulated ones, tracking duplicates
        concepts_cl, new_repeated = data_utils.merge_concepts(concepts_cl, selected_concepts)

        # Find the indices of the merged concepts in the global concept set
        concepts_cl_indx = data_utils.find_indices_of_concepts(concepts_cl, concepts)
        
        # Load features for training and validation sets using the activations saved earlier
        with torch.no_grad():
            if args.strategy != "full_rehearsal":
                # For strategies other than "full_rehearsal", only use the current experience data
                target_features = torch.load(target_save_name, map_location="cpu").float()
                target_features = target_features[grouped_train_indices[exp_id],:] # Select current experience's training data

                image_features = torch.load(clip_save_name, map_location="cpu").float()
                image_features = image_features[grouped_train_indices[exp_id],:] # Select current experience's image features
                image_features /= torch.norm(image_features, dim=1, keepdim=True) # Normalize image features

            elif args.strategy == "full_rehearsal":
                # For the "full_rehearsal" strategy, combine data from all previous experiences
                seen_grouped_train_indices = sum([grouped_train_indices[i] for i in range(exp_id + 1)], []) # Combine previous experiences

                target_features = torch.load(target_save_name, map_location="cpu").float() 
                target_features = target_features[seen_grouped_train_indices,:] # Select all previous experience's data

                image_features = torch.load(clip_save_name, map_location="cpu").float()
                image_features = image_features[seen_grouped_train_indices,:] # Select all previous experience's image features
                image_features /= torch.norm(image_features, dim=1, keepdim=True) # Normalize image features

            # Load text features for the selected concepts
            text_features = torch.load(text_save_name, map_location="cpu").float() 
            text_features = text_features[concepts_cl_indx, :] # Select text features for current concepts
            text_features /= torch.norm(text_features, dim=1, keepdim=True) # Normalize text features

            # Compute clip features by taking the dot product between image and text features
            clip_features = image_features @ text_features.T 

            # Load validation data (for evaluation)
            val_target_features = torch.load(val_target_save_name, map_location="cpu").float()
            val_target_features =  val_target_features[grouped_test_indices[exp_id],:] # Select validation target features

            val_image_features = torch.load(val_clip_save_name, map_location="cpu").float()
            val_image_features = val_image_features[grouped_test_indices[exp_id],:] # Select validation image features
            val_image_features /= torch.norm(val_image_features, dim=1, keepdim=True) # Normalize validation image features
            
            # Compute validation clip features
            val_clip_features = val_image_features @ text_features.T

            # Clean up loaded features from memory to avoid using too much GPU memory
            del image_features, text_features, val_image_features
            torch.cuda.empty_cache()
        
        # Map original class labels to the new grouped class targets for both training and validation
        train_targets = data_utils.get_targets_only(d_train)

        if args.strategy != "full_rehearsal":
            train_targets= list(np.array(train_targets)[grouped_train_indices[exp_id]])  # Select training targets for the current experience
        elif  args.strategy == "full_rehearsal":
            seen_grouped_train_indices = sum([grouped_train_indices[i] for i in range(exp_id + 1)], []) # All previous experiences
            train_targets= list(np.array(train_targets)[seen_grouped_train_indices]) # Select all previous training targets
        
        # Map original targets to the new targets using the mapping
        train_targets = [mapping_from_classes_to_cl_classes[target] for target in train_targets]

        # Similarly, handle the validation targets
        val_targets = data_utils.get_targets_only(d_val) # Get original class labels for validation
        val_targets = list(np.array(val_targets)[grouped_test_indices[exp_id]]) # Select validation targets for the current experience
        val_targets = [mapping_from_classes_to_cl_classes[target] for target in val_targets] # Map to the new class targets

        # Print the number of training and validation targets (samples)
        print(f"Number of training samples: {len(train_targets)}")
        print(f"Number of validation samples: {len(val_targets)}")

        # Print the unique class labels for training and validation
        print(f"Unique training class labels: {set(train_targets)}")
        print(f"Unique validation class labels: {set(val_targets)}")

        # Initialize the projection layer
        if exp_id == 0:
            # For the first experience, initialize a new projection layer
            proj_layer = torch.nn.Linear(
                in_features = target_features.shape[1], # Input features from the backbone model
                out_features = len(concepts_cl), # Output features based on the number of current concepts
                bias = False, # No bias in the projection layer
            ).to(args.device)
        else: 
            # For subsequent experiences, reuse the previous projection layer's weights
            prev_weights = proj_layer.weight.data
            proj_layer = torch.nn.Linear(
                in_features = target_features.shape[1], 
                out_features = len(concepts_cl),
                bias = False
            ).to(args.device)
            proj_layer.weight.data[:prev_weights.shape[0], :] = prev_weights.clone()  # Retain previous weights

        # Train the projection layer and retrieve the best weights
        opt = torch.optim.Adam(proj_layer.parameters(), lr=1e-3) # Adam optimizer for the projection layer
        if args.strategy in ["bottleneck_prototype", "backbone_prototype"]  and exp_id > 0:
            # For "bottleneck_prototype" and "backbone_prototype" strategies, use LwF for learning projection layer
            best_weights = training_utils.train_projection_layer_lwf(
                proj_layer = proj_layer, 
                opt = opt, 
                target_features = target_features, 
                clip_features = clip_features, 
                val_target_features = val_target_features, 
                val_clip_features = val_clip_features, 
                similarity_fn = similarity_fn, 
                args = args,
                n_c_prev = prev_weights.shape[0], # Pass the number of previous concepts
            )
        else:
            # For other strategies, train the projection layer without LwF
            best_weights = training_utils.train_projection_layer(
                proj_layer = proj_layer, 
                opt = opt, 
                target_features = target_features, 
                clip_features = clip_features, 
                val_target_features = val_target_features, 
                val_clip_features = val_clip_features, 
                similarity_fn = similarity_fn, 
                args = args
            )
        
        # Load the best weights after training
        proj_layer.load_state_dict({"weight":best_weights})
        del best_weights  # Clean up best weights
        torch.cuda.empty_cache() # Free up GPU memory

        if args.strategy == "backbone_prototype":
            if exp_id == 0:
                # Computes the mean feature vector for each class in the training data
                class_means = training_utils.compute_class_means_incremental(target_features, train_targets, None)
                
            else:
                # Updates the class_means dictionary with new class means.
                class_means = training_utils.compute_class_means_incremental(target_features, train_targets, class_means)
                
                # Finds the closest new class for each previously seen class based on cosine similarity and calculates the difference between their mean feature vectors.
                mean_diff_dict = training_utils.find_nearest_classes(class_means, set(train_targets))
                
                # Extends the training data by adding pseudo features generated from mean differences and updates the training targets accordingly.
                target_features, train_targets = training_utils.extend_train_data(target_features, train_targets, mean_diff_dict)
            

        # Normalize training features and create datasets
        with torch.no_grad():
            # Process the training features through the projection layer in batches
            train_c = training_utils.batch_process(
                features = target_features, 
                proj_layer = proj_layer, 
                batch_size = min(args.proj_batch_size, len(target_features)), 
                device = args.device, 
                n_c_prev = None,
                )
            # Process the validation features through the projection layer in batches
            val_c = training_utils.batch_process(
                features = val_target_features, 
                proj_layer = proj_layer, 
                batch_size = min(args.proj_batch_size, len(val_target_features)),
                device = args.device, 
                n_c_prev = None,
                )
            
            if args.strategy == "bottleneck_prototype":
                if exp_id == 0:
                    # Computes the mean feature vector for each class in the training data
                    class_means = training_utils.compute_class_means_incremental(train_c, train_targets, None) 
                else:
                    # Updates the class_means dictionary with new class means.
                    class_means = training_utils.compute_class_means_incremental(train_c, train_targets, class_means)
                    # Finds the closest new class for each previously seen class based on cosine similarity and calculates the difference between their mean concept vectors.
                    mean_diff_dict = training_utils.find_nearest_classes(class_means, set(train_targets))
                    # Extends the training data by adding pseudo concepts generated from mean differences and updates the training targets accordingly.
                    train_c, train_targets = training_utils.extend_train_data(train_c, train_targets, mean_diff_dict)
                
            
            # Normalize the training features (zero mean, unit variance)
            train_mean = torch.mean(train_c, dim=0, keepdim=True)
            train_std = torch.std(train_c, dim=0, keepdim=True)
            train_c = (train_c - train_mean) / train_std

            # Convert training targets to a tensor
            train_y = torch.LongTensor(train_targets)
            # Create a dataset for the training data
            indexed_train_ds = IndexedTensorDataset(train_c, train_y)

            # Normalize validation features using the training mean and std
            val_c = (val_c - train_mean) / train_std
            val_y = torch.LongTensor(val_targets)
            # Create a dataset for the validation data
            val_ds = TensorDataset(val_c,val_y)

        # Create data loaders for both training and validation datasets
        indexed_train_loader = DataLoader(indexed_train_ds, batch_size=args.saga_batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=args.saga_batch_size, shuffle=False)
        
        # Initialize a linear model for classification and zero-initialize its weights and bias
        if exp_id == 0:
            linear = torch.nn.Linear(train_c.shape[1],len(classes_cl)).to(args.device)
            linear.weight.data.zero_()
            linear.bias.data.zero_()

            # Set up parameters for the GLM-SAGA solver
            STEP_SIZE = args.SAGA_lr
            ALPHA = 0.99
            metadata = {
                'max_reg': {'nongrouped': args.lam},  # Sparsity regularization
                'strategy': args.strategy,
            }

            # Train the model using GLM-SAGA
            output_proj = glm_saga(
                linear = linear, 
                loader = indexed_train_loader, 
                max_lr = STEP_SIZE, 
                nepochs = args.n_iters,
                alpha = ALPHA, 
                epsilon = 1, 
                k = 1, 
                val_loader = val_loader, 
                do_zero = False, 
                metadata = metadata, 
                n_ex = len(train_c), 
                n_classes = len(classes_cl)
            )
        else:
            # Initialize the linear model for subsequent experiences and copy previous weights
            linear = torch.nn.Linear(train_c.shape[1],len(classes_cl)).to(args.device)
            linear.weight.data.zero_()
            linear.bias.data.zero_()

            if args.strategy != "disjoint_pred":
                 # For strategies other than "disjoint_pred", retain the previous weights
                prev_weights = W_g.clone()
                prev_bias = b_g.clone()
                linear.weight.data[:prev_weights.shape[0], :prev_weights.shape[1]] = prev_weights
                linear.bias.data[:prev_bias.shape[0]] = prev_bias
            

            # Set up GLM-SAGA parameters for training
            STEP_SIZE = args.SAGA_lr
            ALPHA = 0.99
            metadata = {
                'max_reg': {'nongrouped': args.lam},
                'strategy': args.strategy,
            }
            
            # Train the model using GLM-SAGA
            output_proj = glm_saga(
                linear = linear, 
                loader = indexed_train_loader, 
                max_lr = STEP_SIZE, 
                nepochs = args.n_iters, 
                alpha = ALPHA, 
                epsilon = 1, 
                k = 1, 
                val_loader = val_loader, 
                do_zero = False, 
                metadata = metadata, 
                n_ex = len(train_c), 
                n_classes = len(classes_cl),
                )
            
        # Update the weights of the linear model with the new results from GLM-SAGA
        W_c = proj_layer.weight
        if args.strategy == "disjoint_pred" and exp_id > 0:
            # For "disjoint_pred" strategy, adjust the weights with previous experience
            prev_weights = W_g.clone() # .weight.data
            prev_bias = b_g.clone() # .weight.data
            W_g = output_proj['path'][0]['weight']
            W_g[:prev_weights.shape[0], :prev_weights.shape[1]] += prev_weights
            b_g = output_proj['path'][0]['bias']
            b_g[:prev_bias.shape[0]] = prev_bias
        else:
            # Otherwise, just assign the new weights from the current experience
            W_g = output_proj['path'][0]['weight']
            b_g = output_proj['path'][0]['bias']

        # Save the updated weights to the linear model
        linear.weight.data = W_g.clone()
        linear.bias.data = b_g.clone()

        # Set up the directory for saving the results of this experience
        save_name = "{}/{}/{}_cbm/{}_backbone_{}_clip_name/{}_seed_{}_nexp/exp_{}".format(args.save_dir, args.strategy, args.dataset, args.backbone, args.clip_name, args.seed, args.n_experiences, exp_id)
        if not os.path.exists(save_name):
            os.makedirs(save_name)
        
        # Save important variables and model weights
        torch.save(train_mean, os.path.join(save_name, "proj_mean.pt"))
        torch.save(train_std, os.path.join(save_name, "proj_std.pt"))
        torch.save(W_c, os.path.join(save_name ,"W_c.pt"))
        torch.save(W_g, os.path.join(save_name, "W_g.pt"))
        torch.save(b_g, os.path.join(save_name, "b_g.pt"))

        # Save the list of concepts
        with open(os.path.join(save_name, "concepts.txt"), 'w') as f:
            f.write(concepts_cl[0])
            for concept in concepts_cl[1:]:
                f.write('\n'+concept)
        
        # Save the arguments used for the experiment
        with open(os.path.join(save_name, "args.txt"), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Save some metrics related to the experiment
        with open(os.path.join(save_name, "metrics.txt"), 'w') as f:
            out_dict = {}
            for key in ('lam', 'lr', 'alpha', 'time'):
                out_dict[key] = float(output_proj['path'][0][key])
            out_dict['metrics'] = output_proj['path'][0]['metrics']
            nnz = (W_g.abs() > 1e-5).sum().item()
            total = W_g.numel()
            out_dict['sparsity'] = {"Non-zero weights":nnz, "Total weights":total, "Percentage non-zero":nnz/total}
            json.dump(out_dict, f, indent=2)

        print(f'Saving for exp {exp_id} finished ')