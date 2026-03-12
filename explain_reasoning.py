import numpy as np
import argparse
import data_utils
from data_utils import DATASET_ROOTS
import cbm
import torch
from torch.utils.data import Subset
import utils
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

import re
import numpy as np
import matplotlib.pyplot as pl
import seaborn as sns
from final_layer_eval import split_classes_efficiently

sns.set(style="whitegrid")

parser = argparse.ArgumentParser(description='Label-Free Concept Bottleneck Model for Class Incremental Learning - Explain reasoning for a specific image')
parser.add_argument("--device", type=str, default="cuda", help="Device to use for computation")
parser.add_argument("--seed", type=int, default=1993, help="Random seed for reproducibility")
parser.add_argument("--backbone", type=str, default="resnet18_FeTrIL_imagenetsubset_b50", help="Pretrained model to use as backbone")
parser.add_argument("--strategy", type=str, default="backbone_prototype", help="Our CL strategy")
parser.add_argument("--dataset", type=str, default="imagenetsubset", help="Dataset name")
parser.add_argument("--n_experiences", type=int, default=5, help="Number of incremental experiences")
parser.add_argument("--save_dir", type=str, default='saved_models', help="Directory for saving trained models")
parser.add_argument("--concept_set", type=str, default=None, help="Path to txt file containing concept set")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", choices=utils.CLIP_MODEL_NAMES, help="CLIP or SigLIP model to use")
parser.add_argument("--half_split", action="store_true", help="If set, the first half of the classes will be used in the first experience, and the second half will be split across the remaining experiences.")
parser.add_argument("--class_id", type=str, required=True, help="Class ID for the image to analyze")
parser.add_argument("--image_name", type=str, required=True, help="Image name with extension (e.g., 'image_001.jpg' or 'image_001.JPEG')")
parser.add_argument("--gt_label", type=str, required=True, help="Ground truth label for the image")
parser.add_argument("--concept_percentage", type=float, default=100.0, help="Percentage of concepts to use (0-100). If < 100, looks in directories with _concepts_{percentage}pct suffix")
args = parser.parse_args()

if args.half_split == True:
    args.n_experiences += 1



import numpy as np
import matplotlib.pyplot as pl
import torch
from PIL import Image
import data_utils
import utils
import cbm

if __name__ == '__main__':
    utils.set_seed(args.seed)
    device = args.device

    _, target_preprocess = data_utils.get_target_model(args.backbone, device)

    val_dataset_name = args.dataset + "_val"
    val_dataset = data_utils.get_data(val_dataset_name, preprocess=target_preprocess)
    val_pil_data = data_utils.get_data(val_dataset_name)

    cls_file = data_utils.LABEL_FILES[args.dataset]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    grouped_classes, mapping_from_classes_to_cl_classes = split_classes_efficiently(
        n_experiences = args.n_experiences, 
        dataset_name = args.dataset, 
        classes = classes,
        half_split = args.half_split,
        seed = args.seed
    )
    print('claasses in the first experience',grouped_classes[0])
    to_display = f"{args.class_id}_{args.image_name}"
    gt_label = [args.gt_label]
    fig, axs = pl.subplots(1, 4, figsize=(16, 4))
    fig.tight_layout(pad=5.0)
    axs = np.array([axs])
    title_fontsize = 15
    label_fontsize = 15
    xlabel_fontsize = 13

    class_id = args.class_id
    image_name = args.image_name

    if args.dataset == "places365":
        img_path = f"data/places365/data_256_standard/{class_id[0]}/{class_id}/{image_name}"
    else:
        dataset_key = f"{args.dataset}_val"
        dataset_folder = DATASET_ROOTS[dataset_key]
        img_path = f"{dataset_folder}/{class_id}/{image_name}"

    raw_img = Image.open(img_path)
    width, height = raw_img.size
    if width > height:
        left = (width - height) // 2
        right = left + height
        top, bottom = 0, height
    else:
        top = (height - width) // 2
        bottom = top + width
        left, right = 0, width

    cropped_img = raw_img.crop((left, top, right, bottom))

    axs[0][0].imshow(cropped_img.resize([320, 320]))
    axs[0][0].axis('off')
    axs[0][0].set_title(f"GT: {gt_label[0].capitalize()}", fontsize=title_fontsize, fontweight='bold', pad=10)
    processed_img = target_preprocess(raw_img).unsqueeze(0).to(device)

    concepts_cl = []
    classes_cl = []

    for exp_id in range(3):
        print("#" * 20, f"experience {exp_id}", "#" * 20)

        classes_cl = classes_cl + grouped_classes[exp_id]
        
        load_dir = "{}/{}/{}_cbm/{}_backbone_{}_clip_name/{}_seed_{}_nexp/exp_{}".format(args.save_dir, args.strategy, args.dataset, args.backbone, args.clip_name, args.seed, args.n_experiences, exp_id)

        if args.concept_percentage < 100.0:
            load_dir = load_dir + f"_concepts_{args.concept_percentage}pct"

        concept_cl_path = os.path.join(load_dir, "concepts.txt")
        if os.path.exists(concept_cl_path):
            with open(concept_cl_path, 'r') as file:
                concepts_cl = [line.strip() for line in file]
        else:
            selected_concepts = data_utils.get_concepts_for_classes(grouped_classes[exp_id], args.dataset)
            concepts_cl, new_repeated = data_utils.merge_concepts(concepts_cl, selected_concepts)
        
        model = cbm.load_cbm(load_dir, device)
        model.eval()

        with torch.no_grad():
            outputs, concept_act = model(processed_img)

            top_logit_vals, top_classes = torch.topk(outputs[0], dim=0, k=2)
            conf = torch.nn.functional.softmax(outputs[0], dim=0)

            print("Image:{} Gt:{}, 1st Pred:{}, {:.3f}, 2nd Pred:{}, {:.3f}".format(to_display, class_id, classes_cl[top_classes[0]], top_logit_vals[0],
                                                                    classes_cl[top_classes[1]], top_logit_vals[1]))

            contributions = concept_act[0] * model.final.weight[top_classes[0], :]
            num_concepts = min(len(concepts_cl), concept_act[0].shape[0])
            feature_names = [("NOT " if concept_act[0][i] < 0 else "") + concepts_cl[i] for i in range(num_concepts)]
            values = contributions.cpu().numpy()

            max_display = 7
            values_to_use = values[:num_concepts]
            top_indices = np.argsort(values_to_use)[-max_display:]

            labels = [feature_names[i] for i in top_indices]
            print("labels", labels)
            top_values = values_to_use[top_indices]

            other_contributions_sum = np.sum(np.delete(values_to_use, top_indices))
            other_contribution_label = f'Sum of {len(feature_names) - max_display} other features'

            all_values = np.append(top_values[::-1], other_contributions_sum)
            all_labels = labels[::-1] +  [other_contribution_label]

            colors = ['#FF0000' if 'NOT ' in label else '#1F75FE' for label in labels[::-1]] + ['#D3D3D3']

            axs[0][exp_id + 1].barh(all_labels[::-1], all_values[::-1], color=colors[::-1], edgecolor='black', height=0.75)
            axs[0][exp_id + 1].set_xlabel('Contribution Value', fontsize=xlabel_fontsize)
            pred_class = classes_cl[top_classes[0]].split(',')[0]
            confidence = conf[top_classes[0]]

            if pred_class.capitalize() == gt_label[0].capitalize():
                title_color = 'green'
            else:
                title_color = 'red'

            title_color = 'green'

            axs[0][exp_id + 1].set_title(
                f"Phase {exp_id + 1}\nPred: {pred_class.capitalize()}", 
                fontsize=title_fontsize, color=title_color, fontweight='bold'
            )

            axs[0][exp_id + 1].grid(axis='x', linestyle='--', alpha=0.7)
            axs[0][exp_id + 1].tick_params(axis='y', labelsize=label_fontsize, which='both', labelleft=True, labelright=False)
            axs[0][exp_id + 1].tick_params(axis='x', labelsize=label_fontsize-4)

    pl.subplots_adjust(wspace=1.1, hspace=0.3)
    pl.tight_layout()

    output_dir = "visualizations_results"
    os.makedirs(output_dir, exist_ok=True)

    image_name_clean = image_name.replace('.', '_').replace('/', '_')
    base_filename = f"explain_reasoning_{class_id}_{image_name_clean}"

    output_filename_pdf = os.path.join(output_dir, f"{base_filename}.pdf")
    pl.savefig(output_filename_pdf, bbox_inches='tight', transparent=True, dpi=300)
    print(f"Figure saved as {output_filename_pdf}")

    output_filename_png = os.path.join(output_dir, f"{base_filename}.png")
    pl.savefig(output_filename_png, bbox_inches='tight', dpi=300)
    print(f"Figure saved as {output_filename_png}")

