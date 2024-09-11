"""
This script prepares a folder for a particular run.  It pulls the images from data/images and
data/masks and sorts then into training/validation/testing based on a single pre-arrange sort
found in the text files loaded here.  This is done so that all models are working with exactly
the same configuration of images.
"""

import os
import shutil

git_root_path = "/home/shannon/local/Source/Python/bm_study"
root_path = "/home/shannon/local/Source/Python/bm_study/unet"
run_folder = 'run_unet_images_01'

# Define the source and destination directories
source_images = os.path.join(root_path, "data/images")
source_masks = os.path.join(root_path, "data/masks")

# Make folders
os.makedirs(os.path.join(run_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(run_folder, "train/images"), exist_ok=True)
os.makedirs(os.path.join(run_folder, "train/labels"), exist_ok=True)

os.makedirs(os.path.join(run_folder, "test"), exist_ok=True)
os.makedirs(os.path.join(run_folder, "test/images"), exist_ok=True)
os.makedirs(os.path.join(run_folder, "test/labels"), exist_ok=True)

os.makedirs(os.path.join(run_folder, "valid"), exist_ok=True)
os.makedirs(os.path.join(run_folder, "valid/images"), exist_ok=True)
os.makedirs(os.path.join(run_folder, "valid/labels"), exist_ok=True)

# Crawl through the folders starting with base_folder and delete all png files
for root, dirs, files in os.walk(run_folder):
    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(root, file)
            os.remove(file_path)

# Read the file 'train_image.txt' into a list called train_files
train_files = []
with open(os.path.join(git_root_path, 'train_files.txt'), 'r') as file:
    train_files = file.read().splitlines()
train_files = [file.replace('.tiff', '.png') for file in train_files]

# Read the file 'val_image.txt' into a list called val_files
val_files = []
with open(os.path.join(git_root_path, 'val_files.txt'), 'r') as file:
    val_files = file.read().splitlines()
val_files = [file.replace('.tiff', '.png') for file in val_files]

# Read the file 'test_image.txt' into a list called test_files
test_files = []
with open(os.path.join(git_root_path, 'test_files.txt'), 'r') as file:
    test_files = file.read().splitlines()
test_files = [file.replace('.tiff', '.png') for file in test_files]


# Define the destination directories for each set
train_images_dir = os.path.join(run_folder, "train/images")
train_labels_dir = os.path.join(run_folder, "train/labels")
val_images_dir = os.path.join(run_folder, "valid/images")
val_labels_dir = os.path.join(run_folder, "valid/labels")
test_images_dir = os.path.join(run_folder, "test/images")
test_labels_dir = os.path.join(run_folder, "test/labels")

# Copy the training images and labels to the corresponding directories
for file_name in train_files:
    source = os.path.join(source_images, file_name)
    destination = os.path.join(train_images_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_masks, file_name)
    destination = os.path.join(train_labels_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

# Copy the validation images and labels to the corresponding directories
for file_name in val_files:
    source = os.path.join(source_images, file_name)
    destination = os.path.join(val_images_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_masks, file_name)
    destination = os.path.join(val_labels_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

# Copy the testing images and labels to the corresponding directories
for file_name in test_files:
    source = os.path.join(source_images, file_name)
    destination = os.path.join(test_images_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_masks, file_name)
    destination = os.path.join(test_labels_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")


# Create a yolo yaml file
# yol_file = os.path.join(root_path, run_folder, 'bm.yaml')
# with open(yol_file, 'w') as f:
#     f.write(f'path: ../bm_study/{run_folder}\n')
#     f.write('train: train/images\n')
#     f.write('val: valid/images\n')
#     f.write('test: test/images\n')
#     f.write('names:\n')
#     f.write('  0: \'resorption\'\n')

