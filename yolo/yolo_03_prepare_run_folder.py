"""
This script prepares a set of images for training a model based on images that previous worked well.
(i.e.) semi-supervised learning.
"""

import os
import shutil
# import random

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
from PIL import Image


def delete_pngs(folder):
    # Crawl through the folders starting with base_folder and delete all png files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                os.remove(file_path)

def get_files(file_name):
    with open(file_name, 'r') as file:
        files = file.read().splitlines()
    return [file.replace('.tiff', '.png') for file in files]

git_root_path = "/home/shannon/local/Source/Python/bm_study"
root_path = "/home/shannon/local/Source/Python/bm_study/yolo"
run_folder = 'run_yolo_01'
flip_the_masks = False

# Define the source and destination directories
source_images = os.path.join(root_path, "data/images")
source_masks = os.path.join(root_path, "data/masks")

# Make folders
os.makedirs(os.path.join(root_path, run_folder, "train"), exist_ok=True)
os.makedirs(os.path.join(root_path, run_folder, "train/images"), exist_ok=True)
os.makedirs(os.path.join(root_path, run_folder, "train/labels/resorption"), exist_ok=True)

os.makedirs(os.path.join(root_path, run_folder, "test"), exist_ok=True)
os.makedirs(os.path.join(root_path, run_folder, "test/images"), exist_ok=True)
os.makedirs(os.path.join(root_path, run_folder, "test/labels/resorption"), exist_ok=True)

os.makedirs(os.path.join(root_path, run_folder, "valid"), exist_ok=True)
os.makedirs(os.path.join(root_path, run_folder, "valid/images"), exist_ok=True)
os.makedirs(os.path.join(root_path, run_folder, "valid/labels/resorption"), exist_ok=True)

delete_pngs(os.path.join(root_path, run_folder))

# Read the file 'train_image.txt' into a list called train_files
train_files = get_files(os.path.join(git_root_path, 'train_files.txt'))

# Read the file 'val_image.txt' into a list called val_files
val_files = get_files(os.path.join(git_root_path, 'val_files.txt'))

# Read the file 'test_image.txt' into a list called test_files
test_files = get_files(os.path.join(git_root_path, 'test_files.txt'))

# # Get a list of the png files in source_images
# png_files = [file for file in os.listdir(source_images) if file.endswith('.png')]

# # Define a list of filenames that should be excluded
# exclude = ['A8_A.png', 'E2_C.png', 'H2_B.png', '1899-288-512_RMand_Ext_B4_4.png', '1899-288-512_RMand_Ext_D15_4.png']

# # Remove the exclude images from png_files
# png_files = [file for file in png_files if file not in exclude]

# # Split the png_files into training, validation, and testing sets
# train_files, test_files = train_test_split(png_files, test_size=0.2, random_state=42)
# train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

# Define the destination directories for each set
train_images_dir = os.path.join(run_folder, "train/images")
train_labels_dir = os.path.join(run_folder, "train/labels/resorption")
val_images_dir = os.path.join(run_folder, "valid/images")
val_labels_dir = os.path.join(run_folder, "valid/labels/resorption")
test_images_dir = os.path.join(run_folder, "test/images")
test_labels_dir = os.path.join(run_folder, "test/labels/resorption")

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
yol_file = os.path.join(root_path, run_folder, 'bm.yaml')
with open(yol_file, 'w') as f:
    f.write(f'path: ../bm_study/yolo/{run_folder}\n')
    f.write('train: train/images\n')
    f.write('val: valid/images\n')
    f.write('test: test/images\n')
    f.write('names:\n')
    f.write('  0: \'resorption\'\n')

