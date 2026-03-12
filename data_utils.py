import os
from torchvision import datasets, transforms, models
import numpy as np 
import json
import clip 
import torch 

LABEL_FILES = {
    "cifar10": "data/cifar10_classes.txt",
    "cifar100": "data/cifar100_classes.txt",
    "cub": "data/cub_classes.txt",
    "places365" : "data/categories_places365_clean.txt",
    "tiny_imagenet" : "data/tiny_imagenet_classes.txt",
    "imagenetsubset" : "data/imagenetsubset_classes.txt",
    "imagenet" : "data/imagenet_classes.txt",
    }

# =============================================================================
# IMPORTANT: Edit these paths before running the code!
# Replace YOUR_PATH_TO_DATASET and YOUR_PATH_TO_MODEL with your actual paths.
# =============================================================================

DATASET_ROOTS = {
    "imagenet_train": "YOUR_PATH_TO_DATASET/ImageNet/train/",
    "imagenet_val": "YOUR_PATH_TO_DATASET/ImageNet/validation/",
    "cub_train": "YOUR_PATH_TO_DATASET/CUB/train",
    "cub_val": "YOUR_PATH_TO_DATASET/CUB/test",
    "tiny_imagenet_train": "YOUR_PATH_TO_DATASET/tiny-imagenet-200/train",
    "tiny_imagenet_val": "YOUR_PATH_TO_DATASET/tiny-imagenet-200/val",
    "imagenetsubset_train": "YOUR_PATH_TO_DATASET/seed_1993_subset_100_imagenet/data/train",
    "imagenetsubset_val": "YOUR_PATH_TO_DATASET/seed_1993_subset_100_imagenet/data/val",
}

MODEL_ROOTS = {
    "resnet18_places365": "backbone_checkpoints/resnet18_places365.pth.tar",
    "resnet18_FeTrIL": "backbone_checkpoints/Models_Trained_by_FeTrIL",
    "SelfPromptDeit_mytiny": "backbone_checkpoints/Models_Trained_by_APG",
}

def get_resnet_imagenet_preprocess():
    # Defines the preprocessing transformations required for ResNet on ImageNet images.

    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess

def get_data(dataset_name, preprocess=None):
    """
    Load a dataset using torchvision.datasets.

    Args:
    dataset_name (str): Name of the dataset.
    preprocess (callable, optional): Optional transform to be applied to the data.

    Returns:
    torchvision.datasets.Dataset: Loaded dataset object.
    """

    # Load CIFAR-10 training or validation dataset
    if dataset_name == "cifar10_train":
        dataset = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)   
    elif dataset_name == "cifar10_val":
        dataset = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
    # Load CIFAR-100 training or validation dataset
    elif dataset_name == "cifar100_train":
        dataset = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)   
    elif dataset_name == "cifar100_val":
        dataset = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)
    # Load dataset from local directories defined in DATASET_ROOTS
    elif dataset_name in DATASET_ROOTS.keys():
        dataset = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
    # Load Places365 training or validation dataset with error handling for download issues
    elif dataset_name == "places365_train":
        try:
            # dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
            #                            transform=preprocess)
            dataset = datasets.Places365(root=os.path.join('data', 'places365'), split='train-standard', small=True, download=True,
                                       transform=preprocess)
        except(RuntimeError):
            # dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
            #                        transform=preprocess)
            dataset = datasets.Places365(root=os.path.join('data', 'places365'), split='train-standard', small=True, download=False,
                                   transform=preprocess)    
    elif dataset_name == "places365_val":
        try:
            # dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
            #                        transform=preprocess)
            dataset = datasets.Places365(root=os.path.join('data', 'places365'), split='val', small=True, download=True,
                                   transform=preprocess)
        except(RuntimeError):
            # dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
            #                        transform=preprocess)
            dataset = datasets.Places365(root=os.path.join('data', 'places365'), split='val', small=True, download=False,
                                   transform=preprocess)
            
    else:
        raise ValueError("Unsupported dataset_name.")

    return dataset

def get_targets_only(dataset_name):
    """
    Retrieve only the target labels from a dataset.

    Args:
    dataset_name (str): Name of the dataset.

    Returns:
    list: List of target labels.
    """
    dataset = get_data(dataset_name)
    return dataset.targets

def split_data_SelfPromptDeit(n_experiences, dataset_name, classes, half_split=False):
    if dataset_name == "cifar100":
        # Load CIFAR100 dataset for train and test sets
        train_dataset = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=None)
        test_dataset  = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=None)
    
    elif dataset_name == "imagenetsubset":
        # Load ImageNet subset dataset for train and test sets
        train_dataset = datasets.ImageFolder(DATASET_ROOTS["imagenetsubset_train"], None)
        test_dataset = datasets.ImageFolder(DATASET_ROOTS["imagenetsubset_val"], None)

    else:
        # Raise error if dataset name is unsupported
        raise ValueError("Unsupported dataset_name.")

    train_class_indices = {i: [] for i in range(len(classes))}
    for idx, label in enumerate(train_dataset.targets):
        train_class_indices[label].append(idx)

    test_class_indices = {i: [] for i in range(len(classes))}
    for idx, label in enumerate(test_dataset.targets):
        test_class_indices[label].append(idx)

    shuffled_class_indices = [68, 56, 78, 8, 23, 84, 90, 65, 74, 76, 40, 89, 3, 92, 55, 9, 26, 80, 43, 38, 58, 70, 77, 1, 85, 19, 17, 50, 28, 53, 13, 81, 45, 82, 6, 59, 83, 16, 15, 44, 91, 41, 72, 60, 79, 52, 20, 10, 31, 54, 37, 95, 14, 71, 96, 98, 97, 2, 64, 66, 42, 22, 35, 86, 24, 34, 87, 21, 99, 0, 88, 27, 18, 94, 11, 12, 47, 25, 30, 46, 62, 69, 36, 61, 7, 63, 75, 5, 32, 4, 51, 48, 73, 93, 39, 67, 29, 49, 57, 33]

    print("Shuffled classes:", shuffled_class_indices, np.array(classes)[shuffled_class_indices])

    mapping_from_classes_to_cl_classes = {}
    mapping_from_cl_classes_to_classes = {}
    for i in range(len(shuffled_class_indices)):
        mapping_from_classes_to_cl_classes[shuffled_class_indices[i]] = i
        mapping_from_cl_classes_to_classes[i] = shuffled_class_indices[i]

    # Split dataset into one large group of classes for the first phase, then split the remaining classes into n_experiences - 1 equal groups.
    if half_split:
        if dataset_name == "cifar100" or dataset_name == "imagenetsubset":
            # Special splitting for specific datasets
            if n_experiences == 6 or n_experiences == 11:
                half_point = 50
                first_half = shuffled_class_indices[:half_point]
                second_half = shuffled_class_indices[half_point:]
                
                # The first half forms one group
                first_group = [first_half]
                
                # Split the second half into n_experiences groups
                second_groups = np.array_split(second_half, n_experiences-1)
                
                # Combine the groups
                groups = first_group + second_groups

            elif n_experiences == 21 or n_experiences == 61:
                # Split into two parts
                first_phase_point = 40
                first_phase = shuffled_class_indices[:first_phase_point]
                rest_phase = shuffled_class_indices[first_phase_point:]

                # The first half forms one group
                first_group = [first_phase]

                # Split the second half into n_experiences groups
                second_groups = np.array_split(rest_phase, n_experiences-1)

                # Combine the groups
                groups = first_group + second_groups

            elif n_experiences == 15:
                first_phase_point = 30
                first_phase = shuffled_class_indices[:first_phase_point]
                rest_phase = shuffled_class_indices[first_phase_point:]
                first_group = [first_phase]
                second_groups = np.array_split(rest_phase, n_experiences-1)
                groups = first_group + second_groups
        else:
            print("dataset is not added in data utils")
            print(dataset_name)
            exit()
    else:
        print("half_split should be True")
    
    # Split classes into groups
    grouped_classes = [list(np.array(classes)[group]) for group in groups]
    print("Grouped classes:", grouped_classes)

    # Collect training indices for each group
    grouped_train_indices = [
        [idx for cls in group for idx in train_class_indices[cls]]
        for group in groups
    ]

    # Collect testing indices for each group
    grouped_test_indices = [
        [idx for cls in group for idx in test_class_indices[cls]]
        for group in groups
    ]
    
    return grouped_classes, grouped_train_indices, grouped_test_indices, mapping_from_classes_to_cl_classes, mapping_from_cl_classes_to_classes


def split_data(n_experiences, dataset_name, classes, half_split=False):
    """
    Split dataset into groups based on classes and shuffle.

    Args:
    n_experiences (int): Number of groups to split into.
    dataset_name (str): Name of the dataset
    classes (list): List of class indices or names.

    Returns:
    tuple: A tuple containing:
           - list: List of grouped classes.
           - list: List of grouped training indices.
           - list: List of grouped testing indices.
           - dict: Mapping from original classes to grouped classes.
    """
    # Load train and test datasets for the specified dataset_name
    # Different datasets are loaded with dataset-specific settings
    if dataset_name == "cifar10":
        # Load CIFAR10 dataset for train and test sets
        train_dataset = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=None)
        test_dataset  = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=None)
    
    elif dataset_name == "cifar100":
        # Load CIFAR100 dataset for train and test sets
        train_dataset = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=None)
        test_dataset  = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=None)
    
    elif dataset_name == "cub": 
        # Load CUB dataset for train and test sets
        train_dataset = datasets.ImageFolder(DATASET_ROOTS["cub_train"], None)
        test_dataset = datasets.ImageFolder(DATASET_ROOTS["cub_val"], None)
    
    elif dataset_name == "tiny_imagenet":
        # Load Tiny ImageNet dataset for train and test sets
        train_dataset = datasets.ImageFolder(DATASET_ROOTS["tiny_imagenet_train"], None)
        test_dataset = datasets.ImageFolder(DATASET_ROOTS["tiny_imagenet_val"], None)
    
    elif dataset_name == "imagenet":
        # Load ImageNet dataset for train and test sets
        train_dataset = datasets.ImageFolder(DATASET_ROOTS["imagenet_train"], None)
        test_dataset = datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], None)
    
    elif dataset_name == "imagenetsubset":
        # Load ImageNet subset dataset for train and test sets
        train_dataset = datasets.ImageFolder(DATASET_ROOTS["imagenetsubset_train"], None)
        test_dataset = datasets.ImageFolder(DATASET_ROOTS["imagenetsubset_val"], None)
    
    elif dataset_name == "places365": 
        # Load Places365 dataset, with error handling for download issues
        try:
            train_dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=True,
                                       transform=None)
        except(RuntimeError):
            train_dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='train-standard', small=True, download=False,
                                   transform=None)
        try:
            test_dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=True,
                                   transform=None)
        except(RuntimeError):
            test_dataset = datasets.Places365(root=os.path.expanduser("~/.cache"), split='val', small=True, download=False,
                                   transform=None)
    else:
        # Raise error if dataset name is unsupported
        raise ValueError("Unsupported dataset_name.")
    
    # Initialize dictionaries to store training and testing indices for each class
    train_class_indices = {i: [] for i in range(len(classes))}
    for idx, label in enumerate(train_dataset.targets):
        train_class_indices[label].append(idx)

    test_class_indices = {i: [] for i in range(len(classes))}
    for idx, label in enumerate(test_dataset.targets):
        test_class_indices[label].append(idx)
    
    # Shuffle the class indices for random groupings
    shuffled_class_indices = list(range(len(classes)))
    np.random.shuffle(shuffled_class_indices)
    print("Shuffled classes:", shuffled_class_indices, np.array(classes)[shuffled_class_indices])

    # Create mappings between the original classes and the newly shuffled groups
    mapping_from_classes_to_cl_classes = {}
    mapping_from_cl_classes_to_classes = {}
    for i in range(len(shuffled_class_indices)):
        mapping_from_classes_to_cl_classes[shuffled_class_indices[i]] = i
        mapping_from_cl_classes_to_classes[i] = shuffled_class_indices[i]

    # Split dataset into one large group of classes for the first phase, then split the remaining classes into n_experiences - 1 equal groups.
    if half_split:
        if dataset_name == "cifar100" or dataset_name == "imagenetsubset":
            # Special splitting for specific datasets
            if n_experiences == 6 or n_experiences == 11:
                half_point = 50
                first_half = shuffled_class_indices[:half_point]
                second_half = shuffled_class_indices[half_point:]
                
                # The first half forms one group
                first_group = [first_half]
                
                # Split the second half into n_experiences groups
                second_groups = np.array_split(second_half, n_experiences-1)
                
                # Combine the groups
                groups = first_group + second_groups

            elif n_experiences == 21 or n_experiences == 61:
                # Split into two parts
                first_phase_point = 40
                first_phase = shuffled_class_indices[:first_phase_point]
                rest_phase = shuffled_class_indices[first_phase_point:]

                # The first half forms one group
                first_group = [first_phase]

                # Split the second half into n_experiences groups
                second_groups = np.array_split(rest_phase, n_experiences-1)

                # Combine the groups
                groups = first_group + second_groups
            else:
                print("n_experiences is not supported for imagenetsubset")
                exit()

        elif dataset_name == "tiny_imagenet":
            half_point = 100
            first_half = shuffled_class_indices[:half_point]
            second_half = shuffled_class_indices[half_point:]
            
            # The first half forms one group
            first_group = [first_half]
            
            # Split the second half into n_experiences groups
            second_groups = np.array_split(second_half, n_experiences-1)
            
            # Combine the groups
            groups = first_group + second_groups
            
        elif dataset_name == "imagenet":
            half_point = 500
            first_half = shuffled_class_indices[:half_point]
            second_half = shuffled_class_indices[half_point:]

            # The first half forms one group
            first_group = [first_half]

            # Split the second half into n_experiences groups
            second_groups = np.array_split(second_half, n_experiences-1)
            # Combine the groups
            groups = first_group + second_groups

        else:
            print("dataset is not added in data utils")
            print(dataset_name)
            exit()
    else:
        # Regular split into n_experiences groups
        groups = np.array_split(shuffled_class_indices, n_experiences)
        
    # Split classes into groups
    grouped_classes = [list(np.array(classes)[group]) for group in groups]
    print("Grouped classes:", grouped_classes)

    # Collect training indices for each group
    grouped_train_indices = [
        [idx for cls in group for idx in train_class_indices[cls]]
        for group in groups
    ]

    # Collect testing indices for each group
    grouped_test_indices = [
        [idx for cls in group for idx in test_class_indices[cls]]
        for group in groups
    ]
    
    return grouped_classes, grouped_train_indices, grouped_test_indices, mapping_from_classes_to_cl_classes, mapping_from_cl_classes_to_classes


def get_concepts_for_classes(selected_classes, dataset_name, conceptnet_flag = False):
    """
    Retrieve a list of unique concepts for the given classes from a pre-saved JSON file.

    Args:
    selected_classes (list): List of class names for which concepts are to be retrieved.
    dataset_name (str): Name of the dataset.

    Returns:
    list: List of unique concepts for the given classes.
    """
    # Construct the path to the JSON file containing the concept sets
    if conceptnet_flag == True:
        saved_path = os.path.join('data', 'concept_sets', 'conceptnet', 'filtered_concepts', dataset_name, 'similar_concept_filtered_feature_dict.json')
    else:
        saved_path = os.path.join('data', 'concept_sets', 'filtered_concepts', dataset_name, 'similar_concept_filtered_feature_dict.json')
    
    # Load concept sets from JSON
    with open(saved_path, "r") as f:
        concept_set = json.load(f)
    
    # Collect and deduplicate concepts for the selected classes while preserving order
    selected_concepts = []
    seen_concepts = set()
    for cls in selected_classes:
        for concept in concept_set[cls]:
            if concept not in seen_concepts:
                selected_concepts.append(concept)
                seen_concepts.add(concept)
    
    return selected_concepts

def merge_concepts(list1, list2):
    """
    Merge two lists of concepts while maintaining order and identifying repeated concepts.

    Args:
    list1 (list): First list of concepts.
    list2 (list): Second list of concepts to be merged with list1.

    Returns:
    tuple: A tuple containing:
           - list: Merged list of unique concepts from list1 and list2.
           - set: Set of repeated concepts found in list2.
    """
    # Convert list1 to a set to track seen elements
    seen = set(list1)
    repeated = set()

    # Iterate over list2 and add unique elements to list1 and repeated set
    for item in list2:
        if item not in seen:
            list1.append(item)
            seen.add(item)
        else:
            repeated.add(item)
            
    return list1, repeated

def find_indices_of_concepts(concepts_cl, concepts):
    """
    Find indices of elements from concepts_cl in concepts list.

    Args:
    concepts_cl (list): List of concepts to find indices for.
    concepts (list): List to search for indices in.

    Returns:
    list: List of indices corresponding to each concept in concepts_cl found in concepts.
    """
    indices = [concepts.index(item) for item in concepts_cl]
    return indices

def get_target_model(target_name, device):
    """
    Load and return the appropriate model and preprocessing function based on the provided target name.

    Args:
    target_name (str): Name of the target model.

    Returns:
    tuple: A tuple containing:
        - target_model (torch.nn.Module): The target model loaded on the specified device.
        - preprocess (callable): The preprocessing function for the model.
    """

    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load(MODEL_ROOTS["resnet18_places365"])['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    
    elif target_name.startswith('resnet18_FeTrIL'): # 'resnet18_FeTrIL_cifar100_b50'
        # Extracting dataset name and number of base classes
        parts = target_name.split('_')
        dataset = parts[-2]
        num_base_classes = parts[-1] 
        
        # Load the appropriate model
        model_path = f'{MODEL_ROOTS["resnet18_FeTrIL"]}/{dataset}/seed1993/{num_base_classes}/scratch.pth'
        target_model = torch.load(model_path, weights_only=False)
        if dataset == "imagenet" and isinstance(target_model, dict):
            target_model = models.resnet18(pretrained=False, num_classes=1000).to(device)
            # Load the state_dict from the saved file
            state_dict = torch.load(model_path)
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Remove 'model.' prefix from the state_dict keys
            new_state_dict = {}
            for key in state_dict:
                new_key = key.replace("model.", "")  # Remove the 'model.' prefix
                new_state_dict[new_key] = state_dict[key]

            target_model.load_state_dict(new_state_dict)

        # target_model = torch.load(model_path)
        target_model = target_model.to(device)
        target_model.eval()
        
        # Set the dataset mean and std based on the dataset ()
        if dataset == 'cifar100':
            dataset_mean, dataset_std = [0.5356, 0.4898, 0.4255], [0.2007, 0.1999, 0.1992]
        elif dataset == 'tinyimagenet':
            dataset_mean, dataset_std = [0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]
        elif dataset == 'imagenet':
            dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        elif dataset == 'imagenetsubset':
            dataset_mean, dataset_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        # Preprocess the data
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=dataset_mean, std=dataset_std)
        ])

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    elif target_name == "ViT-B/16-IN21K":
        from transformers import ViTModel, ViTImageProcessor
        target_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        preprocor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        def preprocess(image):
            # Preprocess each image individually (returns a dictionary with 'pixel_values')
            return preprocor(images=image, return_tensors="pt")["pixel_values"][0] 
    
    elif target_name.startswith('SelfPromptDeit_mytiny'): # SelfPromptDeit_mytiny_cifar100_b50 
        parts = target_name.split('_')
        dataset = parts[-2]
        from create_model_for_APG import create_SelfPromptDeit_mytiny
        target_model = create_SelfPromptDeit_mytiny(dataset).to(device)
        test_transforms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
        common_transforms = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        preprocess = transforms.Compose([*test_transforms, *common_transforms])

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess