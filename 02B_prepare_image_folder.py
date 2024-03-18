import os
import shutil
from _shared_code import delete_images
import random
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

base_folder = 'run_03'

# Define the source and destination directories
source_images = "/home/shannon/local/Source/Python/bm_study/images/originals/resized"
source_masks = "/home/shannon/local/Source/Python/bm_study/images/masks/flood_filled"
run_labels = base_folder + '/train/label'
run_images = base_folder + '/train/image'

aug_images = base_folder + '/train/aug'
os.makedirs(aug_images, exist_ok=True)
delete_images(aug_images)

# # Create the run1 directory if it doesn't exist
# os.makedirs(base_folder, exist_ok=True)

# # Create the run1/train directory if it doesn't exist
# if not os.path.exists(base_folder + '/train'):
#     os.makedirs(base_folder + '/train')

# # Create the run1/train/labels directory if it doesn't exist  
# if not os.path.exists(run_labels):
os.makedirs(run_labels, exist_ok=True)

# Create the run1/train/images directory if it doesn't exist
# if not os.path.exists(run_images):
os.makedirs(run_images, exist_ok=True)

delete_images(run_labels)
delete_images(run_images)

# Define a list of filenames that should be excluded
exclude = ['A8_A.png', 'E2_C.png', 'H2_B.png', '1899-288-512_RMand_Ext_B4_4.png', '1899-288-512_RMand_Ext_D15_4.png']
best_images = ['B4_4', 'D13_1', 'D15_2', 'E2_4', 'F17_1', 'A2_D', 'A3_D', 'A4_C', 'A5_C', 'B5_D', 'B6_A', 'C3_A', 'C3_D',
               'D4_B', 'D4_D', 'D7_B', 'D7_D', 'D8', 'E4_D', 'E5_AB', 'E6_D', 'E8_C', 'E8_D', 'F4_C', 'G3_D', 'G4_B', 'G5_B', 'G6_B',
               'G6_D', 'G7_B', 'G7_C', 'H2_D', 'H3_D', 'H6_B']
best_images = [f"{image}.png" for image in best_images]
 
# Copy masks and images from source to run_images and run_labels
for file_name in os.listdir(source_images):
    if file_name.endswith('.png') and file_name not in exclude:
        for good_ones in best_images:
            if file_name.endswith(good_ones):
                source = os.path.join(source_images, file_name)
                destination = os.path.join(run_images, file_name)
                shutil.copyfile(source, destination)
                print(f"Copying {file_name} to {destination}")

                source = os.path.join(source_masks, file_name)
                destination = os.path.join(run_labels, file_name)
                shutil.copyfile(source, destination)
                print(f"Copying {file_name} to {destination}")
                break

# make sure there are the same number of images and labels
num_images = len(os.listdir(run_images))
num_labels = len(os.listdir(run_labels))
if num_images != num_labels:
    print(f"Error: {num_images} images and {num_labels} labels")
    exit()

# make sure the names of the files are the same
images = os.listdir(run_images)
labels = os.listdir(run_labels)

# sort images and labels
images.sort()
labels.sort()

for i in range(num_images):
    if images[i] != labels[i]:
        print(f"Error: {images[i]} and {labels[i]} are not the same")
        exit()  


"""
Pull off 20% of the images from the training set and move them to a test set.
"""

# Define the destination directory for test images
run_validation = base_folder + '/test'

# Create the run1/test directory if it doesn't exist
# if not os.path.exists(run_test):
os.makedirs(run_validation, exist_ok=True)

delete_images(run_validation)

# List the images in run1/images
images = os.listdir(run_images)

# Randomly select 20% of the images
num_images = len(images)
num_validation_images = int(num_images * 0.2)
validation_images = random.sample(images, num_validation_images)

# Move the selected test images to run1/test
for image in validation_images:
    source = os.path.join(run_images, image)
    destination = os.path.join(run_validation, image)
    shutil.move(source, destination)
    source = os.path.join(run_labels, image)
    destination = os.path.splitext(destination)[0] + '_actual' + os.path.splitext(destination)[1]
    shutil.move(source, destination)
    print(f"Moving {image} to {destination}")


"""
Pull off another 30% of the images from the training set and move them a validation set.
"""

# # Define the destination directory for test images
# run_validation = base_folder + '/validation'
# run_valid_images = run_validation + '/image'
# run_valid_labels = run_validation + '/label'

# # Create the run1/test directory if it doesn't exist
# # if not os.path.exists(run_test):
# os.makedirs(run_valid_images, exist_ok=True)
# os.makedirs(run_valid_labels, exist_ok=True)

# delete_images(run_valid_images)
# delete_images(run_valid_labels)

# # List the images in run1/images
# images = os.listdir(run_images)

# # Randomly select 30% of the images
# num_images = len(images)
# num_validation_images = int(num_images * 0.3)
# validation_images = random.sample(images, num_validation_images)

# # Move the selected test images to run1/test
# for image in validation_images:
#     source = os.path.join(run_images, image)
#     destination = os.path.join(run_valid_images, image)
#     shutil.move(source, destination)
#     print(f"Moving {image} to {destination}")

#     source = os.path.join(run_labels, image)
#     destination = os.path.join(run_valid_labels, image)
#     shutil.move(source, destination)
#     print(f"Moving {image} to {destination}")


"""
Now, using this create a data generator
and then augment the data
"""



# data_gen_args = dict(rotation_range=0.2,
#                      width_shift_range=0.05,
#                      height_shift_range=0.05,
#                      shear_range=0.05,
#                      zoom_range=0.05,
#                      horizontal_flip=True,
#                      fill_mode='nearest')
# myGenerator = trainGenerator(20, base_folder + '/train', 'image', 'label', data_gen_args, save_to_dir=augmented_images)

# # you will see 60 transformed images and their masks in data/membrane/train/aug
# num_batch = 3
# for i, batch in enumerate(myGenerator):
#     if i >= num_batch:
#         break
