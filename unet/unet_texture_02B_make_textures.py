"""
This code takes the already downsized and ready for analysis images
and analyzes their texture using GLCM.  Five outputs are written.
In the subsequent steps, three of these are fed into the model.

Note that no steps have been taken to optimize this code, and 
it can take many hours to run.  It would be good to convert it
to parallel processing.
"""

from skimage.io import imread
# from skimage.io import imshow
from skimage.feature import graycomatrix, graycoprops
import os
import numpy as np
import matplotlib.pyplot as plt

root_path = "/home/shannon/local/Source/Python/bm_study/unet"
source_folder = "data/images"
dest_folder = "data/glcm"

os.makedirs(os.path.join(root_path, dest_folder), exist_ok=True)
# SOURCE_PATH = '/home/shannon/local/Source/Python/bm_study/images/originals/resized/'
# DEST_PATH = '/home/shannon/local/Source/Python/bm_study/images/originals/resized/glcm/'

window_size = 7

# This code implements the code on this webpage:
# https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_glcm.html

def glcm_image(img, window = 7):
    """TODO: allow different window sizes by parameterizing 3, 4. Also should
    parameterize direction vector [1] [0]"""
    # texture = np.zeros_like(img)
    textures_size = (5, img.shape[0], img.shape[1])
    textures = np.zeros(textures_size)

    # quadratic looping in python w/o vectorized routine, yuck!
    for i in range(img.shape[0] ):  
        for j in range(img.shape[1] ):  
          
            # don't calculate at edges
            if (i < window // 2 ) or \
               (i > (img.shape[0])) or \
               (j < window // 2) or \
               (j > (img.shape[0] - window // 2 - 1)):          
                continue  
        
            # calculate glcm matrix for 7 x 7 window, use dissimilarity (can swap in
            # contrast, etc.)
            glcm_window = img[i-window // 2: i + window // 2 + 1, j - window // 2 : j + window // 2 + 1]  
            glcm = graycomatrix(glcm_window, [1], [0],  symmetric = True, normed = True )   
            textures[0, i, j] = graycoprops(glcm, 'dissimilarity').item(0)
            textures[1, i, j] = graycoprops(glcm, 'contrast').item(0)
            textures[2, i, j] = graycoprops(glcm, 'homogeneity').item(0)
            textures[3, i, j] = graycoprops(glcm, 'energy').item(0)
            textures[4, i, j] = graycoprops(glcm, 'correlation').item(0)

    return textures

# Get a list of all PNG files in the TRAIN_PATH directory
png_files = [f for f in os.listdir(os.path.join(root_path, source_folder)) if f.endswith('.png') and not 'glcm' in f]

# Loop through the PNG files
for file in png_files:
    # Load the image
    img = imread(os.path.join(root_path, source_folder, file))
    
    # Calculate the glcm_image
    result = glcm_image(img, window=window_size)    
    
    # Get the filename without extension
    filename = os.path.splitext(file)[0]
    
    # Add _glcm to the end of the filename
    new_filename = os.path.join(root_path, dest_folder, filename + '_glcm.png')
    
    # Save the result to the new filename
    plt.imsave(os.path.join(root_path, dest_folder, filename + '_glcm_diss.png'), result[0, :, :], cmap='gray')
    plt.imsave(os.path.join(root_path, dest_folder, filename + '_glcm_cont.png'), result[1, :, :], cmap='gray')
    plt.imsave(os.path.join(root_path, dest_folder, filename + '_glcm_homogen.png'), result[2, :, :], cmap='gray')
    plt.imsave(os.path.join(root_path, dest_folder, filename + '_glcm_energy.png'), result[3, :, :], cmap='gray')
    plt.imsave(os.path.join(root_path, dest_folder, filename + '_glcm_corr.png'), result[4, :, :], cmap='gray')
    
    print(f'Converted {filename} to GLCM layers')

print('Done')
