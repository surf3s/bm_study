'''
This code creates a new resized mask folder.

This first step reads the folder of images and creates a mask for the yellow highlighted areas.
Areas outside the mask are filled with black.  Areas inside the mask are filled with white. 
The mask is saved as a new image in a new folder.
'''

import cv2
import os
from PIL import Image

def delete_images(folder):
    # Crawl through the folders starting with base_folder and delete all png files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                os.remove(file_path)

# Use a windows size of 640 because this is what YOLO uses and we want to use
# the same portion of each image for both models.
window_size = 640

# Resize the images to 256 x 256 pixels because this is the size that UNET expects
image_size = 256

# Path to the folder containing the source images
# These images are not included in the repository
# and so this code is not expected to run as is.
# Adjust these paths accordingly or skip to file 03
# to run the model on the downsamples images already provided here.
root_path = "/home/shannon/local/Source/Python/bm_study"

# Path to the folder containing TIF images with yellow highlighted areas
source_mask_path = "images/masks"

# Path to the folder where the flood filled masking images will be saved
dest_mask_path = "unet/data/masks"
os.makedirs(os.path.join(root_path, dest_mask_path), exist_ok=True)
delete_images(os.path.join(root_path, dest_mask_path))

# Iterate through the files in the folder
for filename in os.listdir(os.path.join(root_path, source_mask_path)):
    if filename.endswith(".tiff"):
        # Load the image
        image_path = os.path.join(root_path, source_mask_path, filename)
        image = cv2.imread(image_path)

        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for yellow color in HSV
        lower_yellow = (25, 250, 250)
        upper_yellow = (35, 255, 255)

        # Threshold the image to get a binary mask of yellow regions
        yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

        height, width = yellow_mask.shape[:2]

        # Perform a flood fill to fill in the area outside the masks
        if yellow_mask[1, 1] == 0:
            cv2.floodFill(yellow_mask, None, (1, 1), 255)
        if yellow_mask[1, width - 1] == 0:
            cv2.floodFill(yellow_mask, None, (width - 1, 1), 255)
        if yellow_mask[height - 1, 1] == 0:
            cv2.floodFill(yellow_mask, None, (1, height - 1), 255)
        if yellow_mask[height - 1, width - 1] == 0:
            cv2.floodFill(yellow_mask, None, (width - 1, height - 1), 255)

        # Invert the mask
        result = cv2.bitwise_not(yellow_mask)

        # This image is a visible check of the mask routine. Save it.
        output_path = os.path.join(root_path, dest_mask_path, os.path.splitext(filename)[0] + ".png")
        image = Image.fromarray(result)

        if window_size:
            # Rescale the image to the base size
            if image.width != 1354 or image.height != 1018:
                image = image.resize((1354, 1018))

            # Extract a window from the center of the image of window_size
            left = (1354 - window_size) // 2
            top = (1018 - window_size) // 2
            right = left + window_size
            bottom = top + window_size
            image = image.crop((left, top, right, bottom))

        if image.width > image_size or image.height > image_size:
            # Resize the image
            image = image.resize((image_size, image_size))

        image.save(output_path)

        print(f"Processed {output_path}, size {image.height} x {image.width}")
