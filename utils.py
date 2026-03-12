import os
import torch
import random
import numpy as np
import math
import clip
from tqdm import tqdm
import data_utils
from torch.utils.data import DataLoader
import open_clip

# Constants
PM_SUFFIX = {"max":"_max", "avg":""}

# Supported CLIP and SigLIP model names (CLIP from openai/CLIP, SigLIP from open_clip)
CLIP_MODEL_NAMES = [
    "ViT-B/32", "ViT-B/16", "ViT-L/14",
    "ViT-L-16-SigLIP-384", "ViT-B-16-SigLIP-384", "ViT-SO400M-14-SigLIP-384",
]


def set_seed(seed):
    """
    Set seed for reproducibility across random number generators.

    Args:
    seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Seed set to {seed}")
    return 


def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    """
    Generate save names for various models and data.

    Args:
    clip_name (str): Name of the CLIP model.
    target_name (str): Name of the target model.
    target_layer (str): Target layer name.
    d_probe (str): Data probe identifier.
    concept_set (str): Path to the concept set.
    pool_mode (str): Pooling mode ('max' or 'avg').
    save_dir (str): Directory to save files.

    Returns:
    tuple: A tuple containing:
           - str: Save name for target model activations.
           - str: Save name for CLIP model activations.
           - str: Save name for text features.
    """
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    elif target_name == "ViT-B/16-IN21K":
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    elif target_name.startswith('SelfPromptDeit_mytiny'):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer, PM_SUFFIX[pool_mode])
    
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name


def _all_saved(save_names):
    """
    Check if all save names exist as files.

    Args:
    save_names (dict): Dictionary of save names.

    Returns:
    bool: True if all save names exist, False otherwise.
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True


def _make_save_dir(save_name):
    """
    Create directory if it doesn't exist.

    Args:
    save_name (str): Full save path.

    Returns:
    bool: True if directory was created, False otherwise.
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        return True
    return False


def save_clip_text_features(model, text, save_name, batch_size=1000):
    """
    Save CLIP model's text features.

    Args:
    model (clip.CLIP): CLIP model.
    text (torch.Tensor): Tensor of text inputs.
    save_name (str): Save path for text features.
    batch_size (int, optional): Batch size for processing. Defaults to 1000.
    """
    if os.path.exists(save_name):
        print(f"CLIP text features loaded from {save_name}")
        return
    
    _make_save_dir(save_name)
    text_features = []

    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))

    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def save_SelfPromptDeit_mytiny_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    if os.path.exists(save_name):
        print(f"SelfPromptDeit_mytiny image features loaded from {save_name}")
        return
    
    _make_save_dir(save_name)
    all_features = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            outputs = model(images.to(device))
            all_features.append(outputs.cpu())
    
    torch.save(torch.cat(all_features), save_name)
    del all_features
    torch.cuda.empty_cache()
    return 

def save_vit_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    if os.path.exists(save_name):
        print(f"VIT image features loaded from {save_name}")
        return
    
    _make_save_dir(save_name)
    all_features = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            # print(images.shape)
            # print(images.shape)
            outputs = model(pixel_values=images.to(device).float())
            # print(images.shape)
            # features = outputs.last_hidden_state.mean(dim=1)
            features = outputs.last_hidden_state[:, 0, :]
            all_features.append(features.cpu())

    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    """
    Save CLIP model's image features.

    Args:
    model (clip.CLIP): CLIP model.
    dataset (torch.utils.data.Dataset): Dataset containing images.
    save_name (str): Save path for image features.
    batch_size (int, optional): Batch size for processing. Defaults to 1000.
    device (str, optional): Device to run inference on. Defaults to "cuda".
    """
    if os.path.exists(save_name):
        print(f"CLIP image features loaded from {save_name}")
        return
    
    _make_save_dir(save_name)
    all_features = []

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    Save activations from target model.

    Args:
    target_model (torch.nn.Module): Target model.
    dataset (torch.utils.data.Dataset): Dataset containing data for activation.
    save_name (str): Save path template for activations.
    target_layers (list, optional): List of target layers to save activations from. Defaults to ["layer4"].
    batch_size (int, optional): Batch size for processing. Defaults to 1000.
    device (str, optional): Device to run inference on. Defaults to "cuda".
    pool_mode (str, optional): Pooling mode ('avg' or 'max'). Defaults to 'avg'.
    """
    _make_save_dir(save_name)
    save_names = {}    
    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    """
    Save activations from CLIP and target models.

    Args:
    clip_name (str): Name of the CLIP model.
    target_name (str): Name of the target model.
    target_layers (list): List of target layers to save activations from.
    d_probe (str): Data probe identifier.
    concept_set (str): Path to the concept set.
    batch_size (int): Batch size for processing.
    device (str): Device to run inference on.
    pool_mode (str): Pooling mode ('avg' or 'max').
    save_dir (str): Directory to save files.
    dataset (str): dataset name
    """
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name, 
                                                                    "{}", d_probe, concept_set, 
                                                                      pool_mode, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}

    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)

    if _all_saved(save_names):
        print('All activations previously saved - using existing activations.')
        return
    
    if "SigLIP" in clip_name:
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_name.split("_")[-1],
                                                                        pretrained="webli",
                                                                        device=device)
    else: 
        clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)

    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)    


    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')

    if "SigLIP" in clip_name:
        text = open_clip.get_tokenizer(clip_name.split("_")[-1])(["{}".format(word) for word in words]).to(device)
    else:
        text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    print("Saving and calculating CLIP's text features")
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    
    print("Saving and calculating CLIP's image features")
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    
    print("Saving and calculating target model's activations")
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    elif target_name == "ViT-B/16-IN21K":
        save_vit_image_features(target_model, data_t, target_save_name, batch_size, device)
    elif target_name.startswith('SelfPromptDeit_mytiny'):
        save_SelfPromptDeit_mytiny_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers, batch_size, device, pool_mode)
    
    return

def get_accuracy_cbm(model, dataset, device, mapping_from_cl_classes_to_classes, batch_size=250, num_workers=2):
    correct = 0
    total = 0

    mapping_tensor = torch.tensor([mapping_from_cl_classes_to_classes[i] for i in range(len(mapping_from_cl_classes_to_classes))], device=device)

    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            mapped_pred = mapping_tensor[pred]
            correct += torch.sum(mapped_pred==labels.to(device)).item()
            total += len(labels)
    return correct/total


def get_cos_similarity(preds, gt, clip_model, mpnet_model, device="cuda", batch_size=200, reduce="mean"):
    """
    preds: predicted concepts, list of strings
    gt: correct concepts, list of strings
    """
    pred_tokens = clip.tokenize(preds).to(device)
    gt_tokens = clip.tokenize(gt).to(device)
    pred_embeds = []
    gt_embeds = []

    #print(preds)
    with torch.no_grad():
        for i in range(math.ceil(len(pred_tokens)/batch_size)):
            pred_embeds.append(clip_model.encode_text(pred_tokens[batch_size*i:batch_size*(i+1)]))
            gt_embeds.append(clip_model.encode_text(gt_tokens[batch_size*i:batch_size*(i+1)]))

        pred_embeds = torch.cat(pred_embeds, dim=0)
        pred_embeds /= pred_embeds.norm(dim=-1, keepdim=True)
        gt_embeds = torch.cat(gt_embeds, dim=0)
        gt_embeds /= gt_embeds.norm(dim=-1, keepdim=True)

    #l2_norm_pred = torch.norm(pred_embeds-gt_embeds, dim=1)
    cos_sim_clip = torch.sum(pred_embeds*gt_embeds, dim=1)

    gt_embeds = mpnet_model.encode([gt_x for gt_x in gt])
    pred_embeds = mpnet_model.encode(preds)
    cos_sim_mpnet = np.sum(pred_embeds*gt_embeds, axis=1)

    if reduce == "mean":
        return float(torch.mean(cos_sim_clip)), float(np.mean(cos_sim_mpnet))
    elif reduce == "none":
        return cos_sim_clip, cos_sim_mpnet
