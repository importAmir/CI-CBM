# wget -O synset_labels.txt https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt

import os
import shutil

# Define the paths
label_file_path = 'synset_labels.txt'
image_directory = './validation'
output_directory = './organized_validation'

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Read the labels from the file
with open(label_file_path, 'r') as file:
    labels = file.read().splitlines()

# Ensure the labels directory exists
unique_labels = set(labels)
for label in unique_labels:
    label_dir = os.path.join(output_directory, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

# Copy images to corresponding label directories
for index, label in enumerate(labels):

    image_filename = f'ILSVRC2012_val_{str(index + 1).zfill(8)}.JPEG'
    src_path = os.path.join(image_directory, image_filename)
    dst_path = os.path.join(output_directory, label, image_filename)
    
    # Check if the source file exists and destination file does not exist before copying
    if os.path.exists(src_path) and not os.path.exists(dst_path):
        shutil.copy(src_path, dst_path)
        print(f"Image {src_path} copied!")
    elif not os.path.exists(src_path):
        print(f"Image {src_path} does not exist!")
    else:
        print(f"Image {dst_path} already exists, skipping copy.")

print("Images have been copied and organized by labels.")
