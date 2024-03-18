import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np
from tensorflow.image import ResizeMethod


# Load the images and masks from the supplied lists of images
def load_data(image_folder, image_filenames, mask_folder, mask_filenames):
    images = []
    masks = []

    for filename in image_filenames:
        img = load_img(os.path.join(image_folder, filename), color_mode='grayscale')
        img = img_to_array(img)
        images.append(img)

    # resize and pad
    images = tf.image.resize_with_pad(images, 128, 128)

    for filename in mask_filenames:
        mask = load_img(os.path.join(mask_folder, filename), color_mode='grayscale')
        mask = img_to_array(mask)
        masks.append(mask)

    # resize and pad
    masks = tf.image.resize_with_pad(masks, 128, 128, method=ResizeMethod.NEAREST_NEIGHBOR)

    images = np.array(images)
    masks = np.array(masks)

    return images, masks


def get_images(folder):
    image_files = []
    for filename in os.listdir(folder):
        if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".bmp") or filename.endswith(".png"):
            image_files.append(filename)
    return image_files


def delete_images(folder):
    for filename in os.listdir(folder):
        if filename.endswith(".tif") or filename.endswith(".tiff") or filename.endswith(".bmp") or filename.endswith(".png"):
            os.remove(os.path.join(folder, filename))
