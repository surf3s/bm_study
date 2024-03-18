import os
import shutil

source_images = "/home/shannon/local/Source/Python/bm_study/images/masks/resized"
source_masks = "/home/shannon/local/Source/Python/bm_study/images/masks/flood_filled"

# Create the new folder
new_folder = "/home/shannon/local/Source/Python/bm_study/review"
os.makedirs(new_folder, exist_ok=True)

# Delete all files in new_folder
for filename in os.listdir(new_folder):
    file_path = os.path.join(new_folder, filename)
    if os.path.isfile(file_path):   
        os.remove(file_path)

# Copy files from source_images to the new folder
for filename in os.listdir(source_images):
    source_path = os.path.join(source_images, filename)
    destination_path = os.path.join(new_folder, filename)
    shutil.copy(source_path, destination_path)

# Copy files from source_masks to the new folder with "_filled" added to the filename
for filename in os.listdir(source_masks):
    source_path = os.path.join(source_masks, filename)
    new_filename = os.path.splitext(filename)[0] + "_filled" + os.path.splitext(filename)[1]
    destination_path = os.path.join(new_folder, new_filename)
    shutil.copy(source_path, destination_path)