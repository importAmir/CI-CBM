import json
import random
import torch
from concept_utils import save_feature_dict, remove_too_long, filter_too_similar_to_cls, filter_too_similar
# import data_utils
# import conceptset_utils

"""
CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
MAX_LEN: max number of characters in a concept

PRINT_PROB: what percentage of filtered concepts will be printed
"""

CLASS_SIM_CUTOFF = 0.85
OTHER_SIM_CUTOFF = 0.9
MAX_LEN = 30
PRINT_PROB = 1

dataset = "imagenetsubset" # cifar10 #  cifar100 # imagenet
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

with open("gpt3_init/gpt3_{}_important.json".format(dataset), "r") as f:
    important_dict = json.load(f)
    print(len(important_dict))
if dataset != "cub":
    with open("gpt3_init/gpt3_{}_superclass.json".format(dataset), "r") as f:
        superclass_dict = json.load(f)
        print(len(superclass_dict))
    with open("gpt3_init/gpt3_{}_around.json".format(dataset), "r") as f:
        around_dict = json.load(f)
        print(len(around_dict))

LABEL_FILES = {"places365":"../categories_places365_clean.txt",
               "imagenet":"../imagenet_classes.txt",
               "cifar10":"../cifar10_classes.txt",
               "cifar100":"../cifar100_classes.txt",
               "cub":"../cub_classes.txt",
               "tiny_imagenet" : "../tiny_imagenet_classes.txt",
               "imagenetsubset" : "../imagenetsubset_classes.txt",
               }

with open(LABEL_FILES[dataset], "r") as f:
    classes = f.read().split("\n")
print(len(classes))

feature_dict = {}
for class_name in classes:
    feature_dict[class_name] = set()

for class_name in classes:
    #print(class_name)
    feature_dict[class_name].update(important_dict[class_name]) 
    if dataset != "cub":
        feature_dict[class_name].update(superclass_dict[class_name]) 
        feature_dict[class_name].update(around_dict[class_name]) 
    feature_dict[class_name] = sorted(list(feature_dict[class_name]))
    
save_feature_dict(feature_dict, dataset, 'initial_feature_dict')

concepts = set()

for values in important_dict.values():
    concepts.update(set(values))

if dataset != "cub":
    for values in superclass_dict.values():
        concepts.update(set(values))
        
    for values in around_dict.values():
        concepts.update(set(values))

print('len(concepts):', len(concepts))

concept_to_class = {} 
for concept in concepts:
    concept_to_class[concept] = set()

for class_name in classes:
    for concept in important_dict[class_name]:
        concept_to_class[concept].update([class_name])
    if dataset != "cub":
        for concept in superclass_dict[class_name]:
            concept_to_class[concept].update([class_name])
        for concept in around_dict[class_name]:
            concept_to_class[concept].update([class_name])

print('filtering too long concepts')
concepts, feature_dict = remove_too_long(concepts, MAX_LEN, feature_dict, concept_to_class, print_prob=1)
save_feature_dict(feature_dict, dataset, 'length_filtered_feature_dict')
print('len(concepts):', len(concepts))


print('filtering similar concepts to classes')
feature_dict, concepts = filter_too_similar_to_cls(concepts, classes, feature_dict, concept_to_class, sim_cutoff=CLASS_SIM_CUTOFF, device=device, print_prob=1)
save_feature_dict(feature_dict, dataset, 'similar_to_classes_filtered_feature_dict')
print('len(concepts):', len(concepts))


print('filtering similar concepts to each other')
feature_dict, concepts =  filter_too_similar(concepts, feature_dict, concept_to_class, sim_cutoff=OTHER_SIM_CUTOFF, device=device, print_prob=1)
for class_name in classes:
    feature_dict[class_name] = sorted(list(set(feature_dict[class_name])))
save_feature_dict(feature_dict, dataset, 'similar_concept_filtered_feature_dict')
print('len(concepts):', len(concepts))

concepts = sorted(list(set(concepts)))
with open("./{}_filtered.txt".format(dataset), "w") as f:
    f.write(concepts[0])
    for concept in concepts[1:]:
        f.write("\n" + concept)