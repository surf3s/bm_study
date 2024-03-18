# from PIL import Image
from PIL import ImageOps
# from tensorflow import keras
# from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import array_to_img
import numpy as np
import os
from _shared_code import load_data
from skimage import io
import tensorflow as tf

# from matplotlib import pyplot as plt


image_source = "/home/shannon/local/Source/Python/bm_study/images/originals/resized"
mask_source = "/home/shannon/local/Source/Python/bm_study/images/masks/binary_mask"
results_path = "/home/shannon/local/Source/Python/bm_study/results"

model = load_model(os.path.join(results_path, 'bm_model.keras'))

# Generate predictions for all images in the validation set
validation_images = np.load(os.path.join(results_path, 'bm_val_images.npy'))

val_dataset = tf.data.Dataset.from_tensor_slices((validation_images))

img_height = 128
img_width = 128


def read_images(img_path):
    img_data = tf.io.read_file(img_path)
    img = tf.io.decode_bmp(img_data)

    return img


def prepare_images(img):
    img = tf.image.resize(img, [img_height, img_width])
    return img


val_dataset = val_dataset.map(read_images)
val_dataset = val_dataset.map(prepare_images)

val_predictions = model.predict(val_dataset)

for i, val_prediction in enumerate(val_predictions):
    mask = np.argmax(val_prediction, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(array_to_img(mask))
    img.save(os.path.join(results_path, "modeled_masks", f"{validation_images[i]}.png"))


# # Display ground-truth target mask
# target_mask = Image.open(os.path.join(mask_source, validation_images[i]))
# target_mask_contrast = ImageOps.autocontrast(target_mask)
# target_mask_contrast.show()

print("Done")
