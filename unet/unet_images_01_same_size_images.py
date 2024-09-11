'''
Prepare images for UNET training by resizing them to 512 x 512 pixels
Store them in a new images folder
'''

import os
from PIL import Image

def delete_images(folder):
    # Crawl through the folders starting with base_folder and delete all png files
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                os.remove(file_path)


def resize(source_path, dest_path, image_size=512, window_size = 0, convert_to_grayscale=False):
    # Path to the subfolder where resized images will be saved
    os.makedirs(dest_path, exist_ok=True)
    delete_images(dest_path)

    # Iterate over each file in the folder
    for filename in os.listdir(source_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # Open the image
            image_path = os.path.join(source_path, filename)
            image = Image.open(image_path)

            resized_image_filename = os.path.join(dest_path, os.path.splitext(filename)[0] + ".png")

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

            image = image.convert("L")  # Make sure it is grayscale
            if not convert_to_grayscale:
                image = image.convert("RGB")    # But convert back to RGB format for some models (e.g. YOLO)

            if image.width > image_size or image.height > image_size:
                # Resize the image
                resized_image = image.resize((image_size, image_size))

                # Save the resized image to the resized folder
                resized_image.save(resized_image_filename)
            else:
                image.save(resized_image_filename)

            # Close the image
            image.close()

            print(f"Resized and saved {filename} as {resized_image_filename}")


# Path to the folder containing the source images
# These images are not included in the repository
# and so this code is not expected to run as is.
# Adjust these paths accordingly or skip to file 03
# to run the model on the downsamples images already provided here.
root_path = "/home/shannon/local/Source/Python/bm_study"
source_path = "images/originals"
dest_path = "unet/data/images"

# Use a windows size of 640 because this is what YOLO uses and we want to use
# the same portion of each image for both models.
window_size = 640

# Resize the images to 256 x 256 pixels because this is the size that UNET expects
image_size = 256

resize(os.path.join(root_path, source_path), os.path.join(root_path, dest_path), image_size=image_size, window_size=window_size, convert_to_grayscale=True)

print("Images resized and saved successfully!")
