# Class-Incremental Concept Bottleneck Models

This repository contains code for training Class-Incremental Concept Bottleneck Models (CI-CBM).  

## Setup
To set up the environment and download necessary datasets and models, follow the instructions below.

1. Install Python (Version 3.7 or higher is recommended.)
2. Install dependencies by running `pip install -r requirements.txt`
3. Install CLIP `pip install git+https://github.com/openai/CLIP.git` and SigLip `pip install open_clip_torch`
3. Download the process datasets by following files in `dataset_download` folder
    * CUB: `bash download_cub.sh`
    * ImageNet Subset: `bash download_imagenetsubset.sh`
    * TinyImgeNet: `bash download_tinyimagenet` and group the validation samples by their class using `tinyimagenet_val_grouping.py`
    * ImageNet: `bash download_imagenet.sh` and group the validation samples by their class using `imagenet_val_grouping.py`
4. Download pretrained backbone
    * Download the ResNet18 model pretrained on Places365: `bash download_rn18_places.sh`
    * Download the ResNet18 models trained from scratch using FeTrIL by running `bash download_rn18_FeTrIL.sh`, or follow the instructions in the FeTrIL repository to train the model on the initial phase of data, which includes about half of the classes.
5. Update the `DATASET_ROOTS` and `MODEL_ROOTS` dictionaries in data_utils.py with the file paths to your datasets and backbone models.

## Running the models

### 1. Train CI-CBM

To train a Class-Incremental Concept Bottleneck Model (CI-CBM), run main.py using the configurations specified in `training_commands.txt`.

Key Parameters:
* `seed`: Random seed for shuffling classes and splitting them into phases (default: 1993).
* `backbone`: Pretrained CNN model, options include resnet18, resnet18_places, or a CNN model trained using FeTrIL. For FeTrIL models, use the format resnet18_FeTrIL_{dataset_name}_b{initial_phase_classes}, e.g., resnet18_FeTrIL_cifar100_b50.
* `strategy`: Strategy for model training (backbone_prototype (default), naive, full_rehearsal).
* `clip_name`: Vision-Language model to use, either SigLip (e.g., ViT-L-16-SigLIP-384 (default)) or CLIP (e.g., ViT-B/16).
* `dataset`: Dataset to train the model on (options: cifar10, cifar100, cub, tiny_imagenet, imagenetsubset, places365).
* `SAGA_lr`: Learning rate for the sparse prediction layer (default: 0.1).
* `n_iters`: Maximum number of iterations for training the sparse prediction layer.
* `lam`: Sparsity regularization parameter.
* `n_experiences`: Number of incremental phases in the experiment.
* `half_split`: If True, use a larger initial phase followed by smaller subsequent phases.
  
### 2. Evaluate trained models

Evaluate the trained models by `running evaluate_cbm.py` with the parameters specified in `training_commands.txt` to calculate performance metrics.

## Sources

CUB dataset: https://www.vision.caltech.edu/datasets/cub_200_2011/

Sparse final layer training: https://github.com/MadryLab/glm_saga

CLIP: https://github.com/openai/CLIP

FeTrIL: https://github.com/GregoirePetit/FeTrIL
