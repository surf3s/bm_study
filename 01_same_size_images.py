import os
from PIL import Image
from _shared_code import delete_images

image_size = 512


def resize(source_path, convert_to_grayscale=False):
    # Path to the subfolder where resized images will be saved
    resized_folder_path = os.path.join(source_path, "resized")
    os.makedirs(resized_folder_path, exist_ok=True)
    delete_images(resized_folder_path)

    # Iterate over each file in the folder
    for filename in os.listdir(source_path):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            # Open the image
            image_path = os.path.join(source_path, filename)
            image = Image.open(image_path)

            resized_image_path = os.path.join(resized_folder_path, os.path.splitext(filename)[0] + ".png")

            if convert_to_grayscale:
                image = image.convert("L")

            if image.width > image_size or image.height > image_size:
                # Resize the image
                resized_image = image.resize((image_size, image_size))

                # Save the resized image to the resized folder
                resized_image.save(resized_image_path)
            else:
                image.save(resized_image_path)

            # Close the image
            image.close()

            print(f"Resized and saved {filename}")


# Path to the folder containing the source images
source_path = "/home/shannon/local/Source/Python/bm_study/images/originals"
resize(source_path, convert_to_grayscale=True)

# Path to the folder containing the mask outlines
# source_path = "/home/shannon/local/Source/Python/bm_study/images/masks"
# resize(source_path, convert_to_grayscale=False)

print("Images resized and saved successfully!")
