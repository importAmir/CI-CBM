import torch
import random

def train_projection_layer(proj_layer, opt, target_features, clip_features, val_target_features,
                           val_clip_features, similarity_fn, args):
    indices = list(range(len(target_features))) # List of indices for target features
    best_val_loss = float("inf") # Track the best validation loss
    best_step = 0 # Track the step corresponding to the best validation loss
    best_weights = None # Store the best projection layer weights
    proj_batch_size = min(args.proj_batch_size, len(target_features)) # Set batch size for projection layer training

    # Loop over the specified number of projection steps
    for i in range(args.proj_steps):
        # Shuffle indices to ensure random sampling for each iteration
        random.shuffle(indices)

        # Train the projection layer in batches
        for start_idx in range(0, len(target_features), proj_batch_size):
            opt.zero_grad() # Reset gradients for the optimizer
            end_idx = min(start_idx + proj_batch_size, len(target_features)) # Get the end index for the batch
            batch_indices = indices[start_idx:end_idx] # Select batch indices
            batch = torch.LongTensor(batch_indices) # Convert indices to tensor
            # Forward pass: compute outputs of the projection layer for the target features
            outs = proj_layer(target_features[batch].to(args.device).detach())
            # Compute the loss (negative similarity between clip features and projection layer outputs)
            loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
            loss = torch.mean(loss) # Average the loss over the batch
            loss.backward() # Backpropagate the loss
            opt.step() # Update the model parameters based on the gradients

        # Evaluate the model every `proj_eval_freq` steps or at the last step
        if i % args.proj_eval_freq == 0 or i == args.proj_steps - 1:
            with torch.no_grad(): # Disable gradient computation during evaluation
                val_loss_list = [] # List to store validation losses

                # Loop over the validation dataset in batches
                for start_idx in range(0, len(val_target_features), proj_batch_size):
                    end_idx = min(start_idx + proj_batch_size, len(val_target_features))
                    val_batch = val_target_features[start_idx:end_idx].to(args.device) # Get a validation batch
                    val_output = proj_layer(val_batch).detach() # Get the output from the projection layer for the batch
                    # Compute the validation loss (negative similarity between validation clip features and outputs)
                    val_loss = -similarity_fn(val_clip_features[start_idx:end_idx].to(args.device).detach(), val_output)
                    val_loss_list.append(torch.mean(val_loss).cpu()) # Append the mean validation loss
                
                # Calculate the average validation loss
                val_loss = sum(val_loss_list) / len(val_loss_list)

            # Update best validation loss and weights if the current loss is lower
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else:
                # Stop if validation loss starts increasing
                print(f"Step: {i}, Avg val similarity: {-val_loss:.4f}, ",
                      f"Best step: {best_step}, Best Avg val similarity: {-best_val_loss:.4f}")
                print('Training stops due to validation loss increase')
                break
            
            # Print progress for every evaluation step
            print(f"Step: {i}, Avg val similarity: {-val_loss:.4f}, ",
                  f"Best step: {best_step}, Best Avg val similarity: {-best_val_loss:.4f}")

    # Clean up memory by deleting variables and freeing GPU memory
    del indices, batch_indices, batch, outs, loss
    del val_batch, val_output, val_loss
    torch.cuda.empty_cache()
    return best_weights # Clear GPU cache to free up memory

def batch_process(features, proj_layer, batch_size, device, n_c_prev):
    # Initialize an empty list to store the processed results
    result_list = []
    with torch.no_grad():
        # Iterate over the features in batches of size `batch_size`
        for start_idx in range(0, len(features), batch_size):
            end_idx = min(start_idx + batch_size, len(features)) # Determine the end index for the current batch
            batch_features = features[start_idx:end_idx].to(device)

            if n_c_prev != None:
                # Process the batch through the projection layer and select only the first `n_c_prev` features
                result_list.append(proj_layer(batch_features)[:, :n_c_prev].detach().cpu())
            else: 
                # Process the batch through the projection layer and store the result
                result_list.append(proj_layer(batch_features).detach().cpu())
    # Delete the batch_features tensor to free memory
    del batch_features

    # Clear the GPU cache to free up memory after processing
    torch.cuda.empty_cache()

    # Concatenate all the results in the list and return the final tensor on the CPU
    return torch.cat(result_list, dim=0).cpu()

def train_projection_layer_lwf(proj_layer, opt, target_features, clip_features, val_target_features,
                           val_clip_features, similarity_fn, args, n_c_prev):
    # Initialize index list and tracking variables for best performance during training
    indices = list(range(len(target_features))) 
    best_val_loss = float("inf") # Store the best validation loss observed
    best_step = 0 # Track the step corresponding to the best validation loss
    best_weights = None # Store the best projection layer weights
    proj_batch_size = min(args.proj_batch_size, len(target_features)) # Determine batch size based on available data
    
    # Precompute features for LwF on the training and validation sets using batch processing
    train_c_lwf = batch_process(target_features, proj_layer, proj_batch_size, args.device, n_c_prev)
    val_c_lwf = batch_process(val_target_features, proj_layer, proj_batch_size, args.device, n_c_prev)

    for i in range(args.proj_steps):
        # Shuffle indices for training data to introduce randomness during training
        random.shuffle(indices)

        for start_idx in range(0, len(target_features), proj_batch_size):
            opt.zero_grad()
            end_idx = min(start_idx + proj_batch_size, len(target_features))
            batch_indices = indices[start_idx:end_idx]
            batch = torch.LongTensor(batch_indices)
            
            # Get the output from the projection layer for the current batch of target features
            outs = proj_layer(target_features[batch].to(args.device).detach())

            # Compute the main loss (contrastive loss between target and clip features)
            loss = -similarity_fn(clip_features[batch].to(args.device), outs)

            # Compute the distillation loss for knowledge transfer from the previous model (LwF)
            dist_loss = -similarity_fn(train_c_lwf[batch].to(args.device), outs[:, :n_c_prev])

            # Combine both losses (main loss + distillation loss) with equal weighting
            loss = 0.5 * torch.mean(loss) + 0.5 * torch.mean(dist_loss)

            # Backpropagate the combined loss
            loss.backward()

            # Update the model weights based on the gradients
            opt.step()
        
        # Validation and model checkpointing: Periodically evaluate the model performance
        if i % args.proj_eval_freq == 0 or i == args.proj_steps - 1:
            with torch.no_grad():
                val_loss_list = []
                val_dist_loss_list = []

                # Evaluate on validation set in batches
                for start_idx in range(0, len(val_target_features), proj_batch_size):
                    end_idx = min(start_idx + proj_batch_size, len(val_target_features))
                    val_batch = val_target_features[start_idx:end_idx].to(args.device)
                    val_output = proj_layer(val_batch).detach()

                    # Calculate the validation loss and distillation loss
                    val_loss = -similarity_fn(val_clip_features[start_idx:end_idx].to(args.device), val_output)
                    val_dist_loss = -similarity_fn(val_c_lwf[start_idx:end_idx].to(args.device), val_output[:, :n_c_prev])

                    # Store average losses for this batch
                    val_loss_list.append(torch.mean(val_loss).cpu())
                    val_dist_loss_list.append(torch.mean(val_dist_loss).cpu())
                
                # Calculate total validation loss by combining both loss terms
                val_loss = sum(val_loss_list) + sum(val_dist_loss_list)
            
             # If validation loss improved, save the model weights and update tracking variables
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else:
                # Stop training early if validation loss increases (indicating overfitting)
                print(f"Step: {i}, Avg val similarity: {-val_loss:.4f}, ",
                      f"Best step: {best_step}, Best Avg val similarity: {-best_val_loss:.4f}")
                print('Training stops due to validation loss increase')
                break
            
            # Print validation progress
            print(f"Step: {i}, Avg val similarity: {-val_loss:.4f}, "
                  f"Best step: {best_step}, Best Avg val similarity: {-best_val_loss:.4f}")
    
    # Cleanup: Free up memory by deleting temporary variables
    del batch_indices, batch, outs, loss, dist_loss
    del val_batch, val_output, val_loss, val_dist_loss

    # Clear CUDA memory cache to avoid memory buildup
    torch.cuda.empty_cache()

    # Return the best model weights found during training
    return best_weights

def train_projection_layer_freeze(proj_layer, opt, target_features, clip_features, val_target_features,
                           val_clip_features, similarity_fn, args, n_c_prev):
    
    # Initialize other variables for training
    indices = list(range(len(target_features))) # Create a list of indices for the training data
    best_val_loss = float("inf") # Track the best validation loss seen so far
    best_step = 0 # The step number corresponding to the best validation loss
    best_weights = None # The best projection layer weights
    proj_batch_size = min(args.proj_batch_size, len(target_features)) # Set the batch size based on arguments

    # Training loop for `args.proj_steps` steps
    for i in range(args.proj_steps):
        random.shuffle(indices) 

        # Process data in batches
        for start_idx in range(0, len(target_features), proj_batch_size):
            opt.zero_grad() # Reset gradients before processing the batch
            end_idx = min(start_idx + proj_batch_size, len(target_features)) # Get the end index for this batch
            batch_indices = indices[start_idx:end_idx] # Get the indices for this batch
            batch = torch.LongTensor(batch_indices) # Convert batch indices to a tensor

            # Get the model's predictions (outputs) for the current batch
            outs = proj_layer(target_features[batch].to(args.device).detach())

            # Calculate the loss (negative similarity between the clip features and projected features)
            loss = -similarity_fn(clip_features[batch].to(args.device).detach(), outs)
            loss = torch.mean(loss) # Average the loss over the batch

            loss.backward() # Backpropagate to calculate gradients

            # Freeze the weights of the first `n_c_prev` output nodes (set their gradients to 0)
            with torch.no_grad():
                proj_layer.weight.grad[:n_c_prev] = 0 # This freezes the gradients of the first `n_c_prev` weights
            
            # Update the weights of the projection layer using the optimizer
            opt.step()

        # Evaluation step: Periodically check the validation loss
        if i % args.proj_eval_freq == 0 or i == args.proj_steps - 1:
            with torch.no_grad(): 
                val_loss_list = []
                # Evaluate on the validation set in batches
                for start_idx in range(0, len(val_target_features), proj_batch_size):
                    end_idx = min(start_idx + proj_batch_size, len(val_target_features))
                    val_batch = val_target_features[start_idx:end_idx].to(args.device)
                    val_output = proj_layer(val_batch).detach() # Get model's output for the validation batch
                    # Calculate the negative similarity loss for this batch
                    val_loss = -similarity_fn(val_clip_features[start_idx:end_idx].to(args.device).detach(), val_output)
                    val_loss_list.append(torch.mean(val_loss).cpu()) # Append the average loss for the batch

                # Compute the total validation loss
                val_loss = sum(val_loss_list) / len(val_loss_list)

            # If the validation loss improves, save the model's weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_step = i
                best_weights = proj_layer.weight.clone()
            else:
                 # Stop training early if validation loss starts increasing (indicating overfitting)
                print(f"Step: {i}, Avg val similarity: {-val_loss:.4f}, ",
                      f"Best step: {best_step}, Best Avg val similarity: {-best_val_loss:.4f}")
                print('Training stops due to validation loss increase')
                break
            # Print the current progress of the training
            print(f"Step: {i}, Avg val similarity: {-val_loss:.4f}, ",
                  f"Best step: {best_step}, Best Avg val similarity: {-best_val_loss:.4f}")

    # Cleanup: Free up memory by deleting temporary variables
    del indices, batch_indices, batch, outs, loss
    del val_batch, val_output, val_loss
    torch.cuda.empty_cache() # Clear GPU memory cache

    return best_weights # Return the best weights found during training


def compute_class_means_incremental(train_c, train_targets, class_means=None):
    """
    Compute the mean of the features in train_c for each class in train_targets and 
    update the existing class_means dictionary with new class means. If a class is 
    already present in class_means, it is skipped.

    Args:
    - train_c (torch.Tensor): A tensor of shape (N, P) or (N, n_c_prev) where N is the number of samples and 
                              P (or n_c_prev) is the number of features.
    - train_targets (list): A list of length N containing the class labels for each sample.
    - class_means (dict, optional): An existing dictionary where the keys are class labels and the values 
                                    are the mean feature vectors for each class.

    Returns:
    - class_means (dict): Updated dictionary with new class means added.
    """
    if class_means is None:
        class_means = {}

    unique_classes = set(train_targets)  # Get unique class labels from the list

    # Calculate mean for each class
    for class_label in unique_classes:
        # Skip if the class has already been processed
        if class_label in class_means:
            print(f"Class {class_label} has already been processed. Skipping.")
            continue

        # Find the indices of samples belonging to the current class
        class_indices = [i for i, target in enumerate(train_targets) if target == class_label]

        # Extract the corresponding rows from train_c
        class_features = train_c[class_indices]

        # Calculate the mean for the class
        class_mean = class_features.mean(dim=0)

        # Store the mean in the dictionary
        class_means[class_label] = class_mean

    return class_means


import torch.nn.functional as F

def find_nearest_classes(class_means, train_targets):
    """
    Find the nearest new class for each previously seen class based on cosine similarity,
    and calculate the difference between their means.

    Args:
    - class_means (dict): A dictionary where the keys are class labels and values are mean feature vectors.
    - train_targets (list): A list of class labels representing new classes in the current phase.

    Returns:
    - mean_diff_dict (dict): A dictionary where the key is the previous class label and the value is a tuple containing:
                             - the difference between the mean of the previous class and the mean of the closest new class,
                             - the closest new class.
    """
    # Identify previous classes (classes that are in class_means but not in the current train_targets)
    previous_classes = [cls for cls in class_means.keys() if cls not in train_targets]

    # New classes are the current classes in train_targets
    new_classes = train_targets
    mean_diff_dict = {}

    for prev_class in previous_classes:
        prev_mean = class_means[prev_class]
        max_similarity = float('-inf')
        closest_class = None

        for new_class in new_classes:
            new_mean = class_means[new_class]

            # Handle different dimensions, compare up to the smallest dimension
            min_dim = min(prev_mean.size(0), new_mean.size(0))
            prev_mean_trimmed = prev_mean[:min_dim]
            new_mean_trimmed = new_mean[:min_dim]

            # Calculate cosine similarity
            cosine_similarity = F.cosine_similarity(prev_mean_trimmed.unsqueeze(0), new_mean_trimmed.unsqueeze(0))

            # Check if this new class is the closest one
            if cosine_similarity > max_similarity:
                max_similarity = cosine_similarity
                closest_class = new_class

        # Once the closest class is found, store the difference between the means and the closest class
        if closest_class is not None:
            mean_diff = torch.zeros_like(class_means[closest_class])
            mean_diff[:min_dim] = prev_mean - class_means[closest_class][:min_dim]
            mean_diff_dict[prev_class] = (mean_diff, closest_class)

    return mean_diff_dict


def extend_train_data(train_c, train_targets, mean_diff_dict):
    """
    Extend the training data by creating pseudo features from mean_diff_dict and updating train_targets.
    Includes both original data and pseudo features.

    Args:
    - train_c (torch.Tensor): A tensor of shape (N, D) where N is the number of samples and D is the feature dimension.
    - train_targets (list): A list of length N containing the class labels for each sample.
    - mean_diff_dict (dict): A dictionary where the key is the previous class label and the value is a tuple containing:
                             - the difference vector,
                             - the closest new class.

    Returns:
    - extended_train_c (torch.Tensor): Extended tensor of shape (N_extended, D) with both original and pseudo features.
    - extended_train_targets (list): Extended list with updated targets corresponding to the new pseudo features.
    """
    # Initialize lists to store extended data
    extended_train_c = [train_c]
    extended_train_targets = train_targets.copy()

    # Process pseudo features for each previous class
    for prev_class, (mean_diff, closest_class) in mean_diff_dict.items():
        # Find all samples from train_c corresponding to the closest class
        indices = [i for i, label in enumerate(train_targets) if label == closest_class]
        if not indices:
            continue  # Skip if there are no samples for the closest class
        
        # Extract features for the closest class
        class_features = train_c[indices]
        
        # Create pseudo features by adding the mean_diff to each feature
        pseudo_features = class_features + mean_diff
        
        # Append pseudo features and corresponding labels to the extended lists
        extended_train_c.append(pseudo_features)
        extended_train_targets += [prev_class] * len(pseudo_features)
    
    # Concatenate all extended features
    extended_train_c = torch.cat(extended_train_c, dim=0)
    
    return extended_train_c, extended_train_targets