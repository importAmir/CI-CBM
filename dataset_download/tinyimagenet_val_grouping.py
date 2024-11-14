import os
import shutil

# Define paths
images_folder = 'val/images'  # Folder containing validation images
annotations_file = 'val/val_annotations.txt'  # Path to the val_annotations.txt file
destination_folder = 'val/'  # Folder to store organized images

# Create destination folder if it doesn't exist
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Read the annotations file
with open(annotations_file, 'r') as file:
    annotations = file.readlines()

# Iterate over each line in the annotations file
for line in annotations:
    # Split the line into parts
    parts = line.strip().split('\t')
    
    # Extract the image file name and its corresponding class
    image_file = parts[0]
    image_class = parts[1]
    
    # Create a folder for the class if it doesn't exist
    class_folder = os.path.join(destination_folder, image_class)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    
    # Source image path
    source = os.path.join(images_folder, image_file)
    
    # Destination image path
    destination = os.path.join(class_folder, image_file)
    
    # Copy the image to the appropriate class folder
    shutil.copy(source, destination)

print("Images have been grouped by class successfully!")
