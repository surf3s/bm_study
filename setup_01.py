'''
Run this file once to get a list of images that will be used across all models
for comparison purposes. The images are split into training, validation, and
testing sets.
'''

import os
from sklearn.model_selection import train_test_split

root_path = "/home/shannon/local/Source/Python/bm_study"    
source_path = "images/originals"

# Get a list of the png files in source_images
tiff_files = [file for file in os.listdir(os.path.join(root_path, source_path)) if file.endswith('.tiff')]

# Define a list of filenames that should be excluded
exclude = ['A8_A.tiff', 'E2_C.tiff', 'H2_B.tiff', '1899-288-512_RMand_Ext_B4_4.tiff', '1899-288-512_RMand_Ext_D15_4.tiff']

# Remove the exclude images from png_files
tiff_files = [file for file in tiff_files if file not in exclude]

# Split the tiff_files into training, validation, and testing sets
train_files, test_files = train_test_split(tiff_files, test_size=0.2, random_state=42)
train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)

# Write the training, validation, and testing sets to files
with open(os.path.join(root_path, 'train_files.txt'), 'w') as f:
    for file in train_files:
        f.write(file + '\n')

with open(os.path.join(root_path, 'val_files.txt'), 'w') as f:
    for file in val_files:
        f.write(file + '\n')

with open(os.path.join(root_path, 'test_files.txt'), 'w') as f:
    for file in test_files:
        f.write(file + '\n')
