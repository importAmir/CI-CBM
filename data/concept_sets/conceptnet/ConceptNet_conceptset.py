import data_utils
import conceptset_utils

"""
ConceptNet params:
LIMIT:how many relations to look up, higher limit -> larger concept set
RELATIONS: which relations to include in search 

filters:
CLASS_SIM_CUTOFF: Concenpts with cos similarity higher than this to any class will be removed
OTHER_SIM_CUTOFF: Concenpts with cos similarity higher than this to another concept will be removed
MAX_LEN: max number of characters in a concept

PRINT_PROB: what percentage of filtered concepts will be printed

"""

LIMIT = 200
RELATIONS = ["HasA", "IsA", "PartOf", "HasProperty", "MadeOf"]#, "AtLocation"]

CLASS_SIM_CUTOFF = 0.85
OTHER_SIM_CUTOFF = 0.9 
MAX_LEN = 30

PRINT_PROB = 0.2

dataset = "cifar10"
save_name = 'data/concept_sets/conceptnet/conceptnet_{}_filtered_new.txt'.format(dataset)

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

cls_file = data_utils.LABEL_FILES[dataset]

with open(cls_file, 'r') as f:
    classes = f.read().split('\n')

feature_dict = conceptset_utils.get_init_conceptnet_dict(classes, LIMIT, RELATIONS)

print(len(feature_dict), feature_dict)
conceptset_utils.save_feature_dict(feature_dict, dataset, 'initial_feature_dict')

concepts = set()

for class_name, class_concepts in feature_dict.items():
    concepts.update(class_concepts)

print("Len of Unique concepts:", len(concepts))

concept_to_class = {} 
for concept in concepts:
    concept_to_class[concept] = set()

for class_name, class_concepts in feature_dict.items():
    for concept in class_concepts:
        concept_to_class[concept].add(class_name)

print("Concept to Class Mapping:", concept_to_class)

print('filtering too long concepts')
concepts, feature_dict = conceptset_utils.remove_too_long(concepts, MAX_LEN, feature_dict, concept_to_class, print_prob=1)
conceptset_utils.save_feature_dict(feature_dict, dataset, 'length_filtered_feature_dict')
print('len(concepts):', len(concepts))

print('filtering similar concepts to classes')
feature_dict, concepts = conceptset_utils.filter_too_similar_to_cls(concepts, classes, feature_dict, concept_to_class, sim_cutoff=CLASS_SIM_CUTOFF, device=device, print_prob=1)
conceptset_utils.save_feature_dict(feature_dict, dataset, 'similar_to_classes_filtered_feature_dict')
print('len(concepts):', len(concepts))


print('filtering similar concepts to each other')
feature_dict, concepts =  conceptset_utils.filter_too_similar(concepts, feature_dict, concept_to_class, sim_cutoff=OTHER_SIM_CUTOFF, device=device, print_prob=1)
for class_name in classes:
    feature_dict[class_name] = sorted(list(set(feature_dict[class_name])))
conceptset_utils.save_feature_dict(feature_dict, dataset, 'similar_concept_filtered_feature_dict')
print('len(concepts):', len(concepts))

concepts = sorted(list(set(concepts)))
with open("data/concept_sets/conceptnet/{}_filtered.txt".format(dataset), "w") as f:
    f.write(concepts[0])
    for concept in concepts[1:]:
        f.write("\n" + concept)