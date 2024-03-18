'''
This first step reads the folder of images and creates a mask for the yellow highlighted areas.
Areas outside the mask are filled with black.  Areas inside the mask are filled with white. 
The mask is saved as a new image in a new folder.
'''

import cv2
import os
from _shared_code import delete_images
from PIL import Image

image_size = 512

# Path to the folder containing TIF images with yellow highlighted areas
source_path = "/home/shannon/local/Source/Python/bm_study/images/masks"

# Path to the folder where the flood filled masking images will be saved
visible_mask_path = "/home/shannon/local/Source/Python/bm_study/images/masks/flood_filled"

# Path to the folder where the binary mask images will be saved
binary_mask_path = "/home/shannon/local/Source/Python/bm_study/images/masks/binary_mask"

delete_images(visible_mask_path)
delete_images(binary_mask_path) 

# Iterate through the files in the folder
for filename in os.listdir(source_path):
    if filename.endswith(".tiff"):
        # Load the image
        image_path = os.path.join(source_path, filename)
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
        output_path = os.path.join(visible_mask_path, os.path.splitext(filename)[0] + ".png")
        img = Image.fromarray(result)
        img = img.resize((image_size, image_size))
        img.save(output_path)
        # cv2.imwrite(output_path, img)

        # Replace all white pixels in the greyscale image result with a value of 1
        result[result == 255] = 1

        # This is the binary mask.  Save it.
        output_path = os.path.join(binary_mask_path, filename)
        cv2.imwrite(output_path, result)

        # Get the width and height of the image
        height, width = result.shape[:2]

        print(f"Processed {filename}, size {height} x {width}")
