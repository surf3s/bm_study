"""
This script prepares a set of images for training a model based on images that previous worked well.
(i.e.) semi-supervised learning.
"""

import os
import shutil
# import random
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import numpy as np
# from PIL import Image
# from sklearn.model_selection import train_test_split

def delete_png_files(folder):
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
root_path = "/home/shannon/local/Source/Python/bm_study/unet"
run_folder = 'run_unet_texture_01'

# Define the source and destination directories
source_images = os.path.join(root_path, "data/images")
source_masks = os.path.join(root_path, "data/masks")
source_texture = os.path.join(root_path, "data/glcm")

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

delete_png_files(os.path.join(root_path, run_folder))

# Read the file 'train_image.txt' into a list called train_files
train_files = get_files(os.path.join(git_root_path, 'train_files.txt'))

# Read the file 'val_image.txt' into a list called val_files
val_files = get_files(os.path.join(git_root_path, 'val_files.txt'))

# Read the file 'test_image.txt' into a list called test_files
test_files = get_files(os.path.join(git_root_path, 'test_files.txt'))

# Define the destination directories for each set
train_images_dir = os.path.join(root_path, run_folder, "train/images")
train_labels_dir = os.path.join(root_path, run_folder, "train/labels")
val_images_dir = os.path.join(root_path, run_folder, "valid/images")
val_labels_dir = os.path.join(root_path, run_folder, "valid/labels")
test_images_dir = os.path.join(root_path, run_folder, "test/images")
test_labels_dir = os.path.join(root_path, run_folder, "test/labels")

# Copy the training images and labels to the corresponding directories
for file_name in train_files:
    source = os.path.join(source_images, file_name)
    destination = os.path.join(train_images_dir, file_name)
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_cont.png'))
    destination = os.path.join(train_images_dir, file_name.replace('.png', '_glcm_cont.png'))
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_diss.png'))
    destination = os.path.join(train_images_dir, file_name.replace('.png', '_glcm_diss.png'))
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_homogen.png'))
    destination = os.path.join(train_images_dir, file_name.replace('.png', '_glcm_homogen.png'))
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

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_cont.png'))
    destination = os.path.join(val_images_dir, file_name.replace('.png', '_glcm_cont.png'))
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_diss.png'))
    destination = os.path.join(val_images_dir, file_name.replace('.png', '_glcm_diss.png'))
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_homogen.png'))
    destination = os.path.join(val_images_dir, file_name.replace('.png', '_glcm_homogen.png'))
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

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_cont.png'))
    destination = os.path.join(test_images_dir, file_name.replace('.png', '_glcm_cont.png'))
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_diss.png'))
    destination = os.path.join(test_images_dir, file_name.replace('.png', '_glcm_diss.png'))
    shutil.copyfile(source, destination)
    print(f"Copying {file_name} to {destination}")

    source = os.path.join(source_texture, file_name.replace('.png', '_glcm_homogen.png'))
    destination = os.path.join(test_images_dir, file_name.replace('.png', '_glcm_homogen.png'))
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


print("Done") 