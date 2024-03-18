import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from sklearn.model_selection import train_test_split
from _shared_code import get_images, load_data
from IPython.display import clear_output
import matplotlib.pyplot as plt
from tensorflow_examples.models.pix2pix import pix2pix


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([train_images[0], train_masks[1], create_mask(model.predict(train_images[0][tf.newaxis, ...]))])


class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions()
        print('\nSample Prediction after epoch {}\n'.format(epoch + 1))


# Load the data
image_source = "/home/shannon/local/Source/Python/bm_study/images/originals/resized"
mask_source = "/home/shannon/local/Source/Python/bm_study/images/masks/binary_mask"
results_path = "/home/shannon/local/Source/Python/bm_study/results"

image_filenames = get_images(image_source)

# Split the data into 80% training and 20% validation
train_image_filenames, val_image_filenames = train_test_split(image_filenames, test_size=0.2, random_state=42)

train_mask_filenames = [os.path.join(mask_source, filename) for filename in train_image_filenames]
train_image_filenames = [os.path.join(image_source, filename) for filename in train_image_filenames]
training_dataset = tf.data.Dataset.from_tensor_slices((train_image_filenames, train_mask_filenames))

val_mask_filesname = [os.path.join(mask_source, filename) for filename in val_image_filenames]
val_image_filenames = [os.path.join(image_source, filename) for filename in val_image_filenames]
val_dataset = tf.data.Dataset.from_tensor_slices((val_image_filenames, val_mask_filesname))


def read_images(img_path, segmentation_mask_path):
    img_data = tf.io.read_file(img_path)
    img = tf.io.decode_bmp(img_data)

    segm_data = tf.io.read_file(segmentation_mask_path)
    segm_mask = tf.io.decode_bmp(segm_data)

    return img, segm_mask


training_dataset = training_dataset.map(read_images)
val_dataset = val_dataset.map(read_images)

# train_images, train_masks = load_data(image_source, train_image_filenames, mask_source, train_mask_filenames)
# val_images, val_masks = load_data(image_source, val_image_filenames, mask_source, val_mask_filenames)

BATCH_SIZE = 4  # could be 64
BUFFER_SIZE = 1000
img_height = 128
img_width = 128


def prepare_images(img, semg_mask):
    img = tf.image.resize(img, [img_height, img_width])
    semg_mask = tf.image.resize(semg_mask, [img_height, img_width], method='nearest')
    return img, semg_mask


training_dataset = training_dataset.map(prepare_images)
training_dataset = training_dataset.batch(BATCH_SIZE)
training_dataset = training_dataset.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

val_dataset = val_dataset.map(prepare_images)
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

TRAIN_LENGTH = 16
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

output_channels = 2

# Define the input shape
input_shape = (128, 128, 1)

base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]


# Define the U-Net model architecture
def unet_model(output_channels : int):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])

    # Downsampling through the model
    skips = down_stack(inputs)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        filters=output_channels, kernel_size=3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


OUTPUT_CLASSES = 2

model = unet_model(output_channels=OUTPUT_CLASSES)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# tf.keras.utils.plot_model(model, show_shapes=True)


EPOCHS = 20
VAL_SUBSPLITS = 5
# VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
VALIDATION_STEPS = 4

# model_history = model.fit(training_dataset, epochs=EPOCHS,
#                           steps_per_epoch=STEPS_PER_EPOCH,
#                           validation_steps=VALIDATION_STEPS,
#                           validation_data=val_dataset)
# # ,
#                           callbacks=[DisplayCallback()])

model_history = model.fit(training_dataset,
                          epochs=EPOCHS,
                        #   steps_per_epoch=STEPS_PER_EPOCH,
                        #   validation_steps=VALIDATION_STEPS,
                          validation_data=val_dataset)


# Save the model
output_path = os.path.join(results_path, 'bm_model.keras')
model.save(output_path)

# Save the history
np.save(os.path.join(results_path, 'bm_history.npy'), model_history.history)

# Save the list of training and validation images
np.save(os.path.join(results_path, 'bm_train_images.npy'), train_image_filenames)
np.save(os.path.join(results_path, 'bm_val_images.npy'), val_image_filenames)

print(f"Model and history saved to {results_path}")
