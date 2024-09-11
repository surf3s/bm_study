import os
from PIL import Image
import numpy as np

# Path to the directory containing the PNG images
root_path = '/home/shannon/local/Source/Python/bm_study/'
image_path = 'images_unet/masks'

# Get the list of PNG files in the directory
png_files = [filename for filename in os.listdir(os.path.join(root_path, image_path)) if filename.endswith('.png')]

results = []
# Loop through the PNG images in the directory
for filename in png_files:
    # Open the image
    image = Image.open(os.path.join(root_path, image_path, filename))

    # Convert the image to numpy array
    pixels = np.array(image).flatten()

    pixels[ pixels > 128 ] = 255
    pixels[ pixels <= 128 ] = 0

    # Calculate the number of white pixels
    white_pixels = np.count_nonzero(pixels)

    # Calculate the ratio of black to white pixels
    ratio = white_pixels / pixels.size

    results.append(ratio)

    # Close the image
    image.close()

# Write the results to a txt file
with open('white_pixels.txt', 'w') as file:
    for ratio in results:
        file.write(f'{ratio}\n')