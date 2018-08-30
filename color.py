from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, SeparableConv2D, UpSampling2D, MaxPooling2D, BatchNormalization, concatenate
from keras.models import Model, Sequential
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
from PIL import Image
import numpy as np
import cv2

run = wandb.init()
config = run.config

config.num_epochs = 10 * 20
config.batch_size = 2
config.img_dir = "images"
config.height = 256
config.width = 256

val_dir = 'test'
train_dir = 'train'
images_per_train_epoch = len(glob.glob(train_dir + "/*"))

# automatically get the data if it doesn't exist
if not os.path.exists("train"):
    print("Downloading flower dataset...")
    subprocess.check_output("curl https://storage.googleapis.com/l2kzone/flowers.tar | tar xz", shell=True)

# TODO preprocess data (resize) and save to disk

def generator(batch_size, img_dir):
    """A generator that returns black and white images and color images.

    We keep only last 2 channels in YCrCb space since Y value is obviously same as gray scale."""

    image_filenames = glob.glob(img_dir + "/*")
    while True:
        bw_images = np.zeros((batch_size, config.width, config.height))
        color_images = np.zeros((batch_size, config.width, config.height, 2))
        random.shuffle(image_filenames)
        for i, img_path in enumerate(random.sample(image_filenames, batch_size)):
            img_BGR = cv2.imread(img_path)
            # TODO apply augmentation (flip, crop, scale) while resizing for test dataset
            img_BGR = cv2.resize(img_BGR, (config.width, config.height))
            img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
            color_images[i] = img_YCrCb[..., 1:] / 127.5 - 1
            bw_images[i] = img_YCrCb[..., 0] / 127.5 - 1
        yield (bw_images, color_images)


# Create model
n_filters = 32
n_layers = 4

skip_layers = []

# First layer
input_gray = Input(shape = (config.height, config.width))  # same as Y channel
CrCb = Reshape((config.height, config.width,1))(input_gray)
CrCb = SeparableConv2D(n_filters, (3, 3), activation='relu', padding='same')(CrCb)
CrCb = BatchNormalization()(CrCb)
CrCb = SeparableConv2D(n_filters, (3, 3), activation='relu', padding='same')(CrCb)
CrCb = BatchNormalization()(CrCb)

# Down layers
for n_layer in range(1, n_layers):
    skip_layers.append(CrCb)
    n_filters *= 2
    CrCb = MaxPooling2D(2,2)(CrCb)
    CrCb = SeparableConv2D(n_filters, (3, 3), activation='relu', padding='same')(CrCb)
    CrCb = BatchNormalization()(CrCb)
    CrCb = SeparableConv2D(n_filters, (3, 3), activation='relu', padding='same')(CrCb)
    CrCb = BatchNormalization()(CrCb)

 # Up layers are made of Transposed convolution + 2 sets of separable convolution
for n_layer in range(1, n_layers):
    n_filters //= 2
    CrCb = UpSampling2D((2, 2))(CrCb)
    CrCb = concatenate([CrCb, skip_layers[-n_layer]], axis = -1)
    CrCb = BatchNormalization()(CrCb)
    CrCb = SeparableConv2D(n_filters, (3, 3), activation='relu', padding='same')(CrCb)
    CrCb = BatchNormalization()(CrCb)
    CrCb = SeparableConv2D(n_filters, (3, 3), activation='relu', padding='same')(CrCb)
    CrCb = BatchNormalization()(CrCb)

# Create output classes
CrCb = Conv2D(2, (1, 1), activation='tanh', padding='same')(CrCb)

model = Model(inputs=input_gray, outputs = CrCb)
model.compile(optimizer='adam', loss='mse')

(val_bw_images, val_color_images) = next(generator(145, val_dir))

model.fit_generator(generator(config.batch_size, train_dir),
                     steps_per_epoch=int(images_per_train_epoch / config.batch_size),
                     epochs=config.num_epochs, callbacks=[WandbCallback(data_type='image', predictions=16),
                     ModelCheckpoint(filepath = 'model/weights.{epoch:03d}.hdf5')],  # TODO add datetime
                     validation_data=(val_bw_images, val_color_images))


