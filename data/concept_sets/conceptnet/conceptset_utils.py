import requests
import random
import numpy as np
import torch
import math
from tqdm import tqdm
import clip
from sentence_transformers import SentenceTransformer    
    
def get_init_conceptnet(classes, limit=200, relations=["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]):
    concepts = set()

    for cls in tqdm(classes):
        words = cls.replace(',', '').split(' ')
        for word in words:
            obj = requests.get('http://api.conceptnet.io/c/en/{}?limit={}'.format(word, limit)).json()
            obj.keys()
            for dicti in obj['edges']:
                rel = dicti['rel']['label']
                try:
                    if dicti['start']['language'] != 'en' or dicti['end']['language'] != 'en':
                        continue
                except(KeyError):
                    continue

                if rel in relations:
                    if rel in ["IsA"]: 
                        concepts.add(dicti['end']['label'])
                    else:
                        concepts.add(dicti['start']['label'])
                        concepts.add(dicti['end']['label'])
    return concepts

from tqdm import tqdm
import requests

def get_init_conceptnet_dict(classes, limit=200, relations=["HasA", "IsA", "PartOf", "HasProperty", "MadeOf", "AtLocation"]):
    class_to_concepts = {}

    for cls in tqdm(classes):
        concepts = set()
        words = cls.replace(',', '').split(' ')
        for word in words:
            obj = requests.get('http://api.conceptnet.io/c/en/{}?limit={}'.format(word, limit)).json()
            obj.keys()
            for dicti in obj['edges']:
                rel = dicti['rel']['label']
                try:
                    if dicti['start']['language'] != 'en' or dicti['end']['language'] != 'en':
                        continue
                except KeyError:
                    continue

                if rel in relations:
                    if rel == "IsA": 
                        concepts.add(dicti['end']['label'])
                    else:
                        concepts.add(dicti['start']['label'])
                        concepts.add(dicti['end']['label'])
        class_to_concepts[cls] = list(concepts)

    return class_to_concepts

import json

import os
def save_feature_dict(feature_dict, dataset, save_name):
    json_object = json.dumps(feature_dict, indent=4)
    dir_path = f"data/concept_sets/conceptnet/filtered_concepts/{dataset}"
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/{save_name}.json", "w") as outfile:
        outfile.write(json_object)
    print(save_name, 'saved')
    return 

def remove_too_long(concepts, max_len, feature_dict, concept_to_class, print_prob=0):
    """
    deletes all concepts longer than max_len
    """
    new_concepts = []
    for concept in concepts:
        if len(concept) <= max_len:
            new_concepts.append(concept)
        else:
            selected_class = concept_to_class[concept]
            for cls in selected_class:
                feature_dict[cls].remove(concept)
            if random.random()<print_prob:
                print(concept, selected_class)
    return new_concepts, feature_dict


def filter_too_similar_to_cls(concepts, classes, feature_dict, concept_to_class, sim_cutoff, device, print_prob=0):
    #first check simple text matches
    concepts = list(concepts)
    concepts = sorted(concepts)
    
    for cls in classes:
        for prefix in ["", "a ", "A ", "an ", "An ", "the ", "The "]:
            try:
                concepts.remove(prefix+cls)
                concept = prefix+cls
                selected_class = concept_to_class[concept]
                for c in selected_class:
                    feature_dict[c].remove(concept)
                if random.random()<print_prob:
                    print("Class:{} - Deleting {} - selected class {}".format(c, concept, selected_class))
            except(ValueError):
                pass
        
        try:
            concepts.remove(cls.upper())
            concept = cls.upper()
            selected_class = concept_to_class[concept]
            for c in selected_class:
                feature_dict[c].remove(concept)
            if random.random()<print_prob:
                print("Class:{} - Deleting {} - selected class {}".format(c, concept, selected_class))
        except(ValueError):
            pass

        try:
            concepts.remove(cls[0].upper()+cls[1:])
            concept = cls[0].upper()+cls[1:]
            selected_class = concept_to_class[concept]
            for c in selected_class:
                feature_dict[c].remove(concept)
            if random.random()<print_prob:
                print("Class:{} - Deleting {} - selected class {}".format(c, concept, selected_class))
        except(ValueError):
            pass
    print(len(concepts))
        
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    class_features_m = mpnet_model.encode(classes)
    concept_features_m = mpnet_model.encode(concepts)
    dot_prods_m = class_features_m @ concept_features_m.T
    dot_prods_c = _clip_dot_prods(classes, concepts, device)
    #weighted since mpnet has highger variance
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    for i in range(len(classes)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff:
                if j not in to_delete:
                    to_delete.append(j)
    concept_list = [concepts[i] for i in to_delete]
    # print("list of deleted concept ", concept_list)
    for concept in concept_list:
        concepts.remove(concept)
        selected_class = concept_to_class[concept]
        for c in selected_class:
            feature_dict[c].remove(concept)
        if random.random()<print_prob:
            print("Deleting {} - selected class {}".format(c, selected_class))
    return feature_dict, concepts

def filter_too_similar(concepts, feature_dict, concept_to_class, sim_cutoff, device="cuda", print_prob=0):
    
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = concept_features @ concept_features.T
    dot_prods_c = _clip_dot_prods(concepts, concepts, device)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    
    to_delete = []
    to_replace = []
    for i in range(len(concepts)):
        for j in range(len(concepts)):
            prod = dot_prods[i,j]
            if prod >= sim_cutoff and i!=j:
                if i not in to_delete and j not in to_delete:
                    #Deletes the concept with lower average similarity to other concepts - idea is to keep more general concepts
                    if np.sum(dot_prods[i]) < np.sum(dot_prods[j]):
                        to_delete.append(i)
                        to_replace.append(j)
                    else:
                        to_delete.append(j)
                        to_replace.append(i)

    concept_list_delete = [concepts[i] for i in to_delete]
    # print("list of deleted concept ", concept_list_delete)
    concept_list_replace = [concepts[i] for i in to_replace]
    # print("list of replace concept ", concept_list_replace)

    to_print = random.random() < print_prob

    for i in range(len(to_delete)):
        deleted_concept = concept_list_delete[i]
        replace_concept = concept_list_replace[i]
        concepts.remove(deleted_concept)
        selected_class = concept_to_class[deleted_concept]
        for c in selected_class:
            feature_dict[c].remove(deleted_concept)
            if replace_concept not in feature_dict[c]:
                feature_dict[c].append(replace_concept)
                concept_to_class[replace_concept].update([c])
        if to_print:
            print("Replacing {} by {} - selected class {} - sim:{:.4f} ".format(deleted_concept, replace_concept, selected_class, dot_prods[to_delete[i],to_replace[i]]))
    return feature_dict, concepts


def _clip_dot_prods(list1, list2, device="cuda", clip_name="ViT-B/16", batch_size=500):
    "Returns: numpy array with dot products"
    clip_model, _ = clip.load(clip_name, device=device)
    text1 = clip.tokenize(list1).to(device)
    text2 = clip.tokenize(list2).to(device)
    
    features1 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text1)/batch_size)):
            features1.append(clip_model.encode_text(text1[batch_size*i:batch_size*(i+1)]))
        features1 = torch.cat(features1, dim=0)
        features1 /= features1.norm(dim=1, keepdim=True)

    features2 = []
    with torch.no_grad():
        for i in range(math.ceil(len(text2)/batch_size)):
            features2.append(clip_model.encode_text(text2[batch_size*i:batch_size*(i+1)]))
        features2 = torch.cat(features2, dim=0)
        features2 /= features2.norm(dim=1, keepdim=True)
        
    dot_prods = features1 @ features2.T
    return dot_prods.cpu().numpy()

def most_similar_concepts(word, concepts, device="cuda"):
    """
    returns most similar words to a given concepts
    """
    mpnet_model = SentenceTransformer('all-mpnet-base-v2')
    word_features = mpnet_model.encode([word])
    concept_features = mpnet_model.encode(concepts)
        
    dot_prods_m = word_features @ concept_features.T
    dot_prods_c = _clip_dot_prods([word], concepts, device)
    
    dot_prods = (dot_prods_m + 3*dot_prods_c)/4
    min_distance, indices = torch.topk(torch.FloatTensor(dot_prods[0]), k=5)
    return [(concepts[indices[i]], min_distance[i]) for i in range(len(min_distance))]