from keras.layers import Input, Dense, Flatten, Reshape, Conv2D, SeparableConv2D, UpSampling2D, MaxPooling2D, BatchNormalization, concatenate
from keras.models import Model, Sequential, load_model
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
import random
import glob
import wandb
from wandb.keras import WandbCallback
import subprocess
import os
import numpy as np
import cv2
from pathlib import Path

run = wandb.init()
config = run.config

config.num_epochs = 100 * 20
config.batch_size = 2
config.img_dir = "images"
config.height = 256
config.width = 256
config.n_layers = 5
config.n_filters = 32
config.crop = 0.15 / 2
config.dataset = 'custom'

model_path = None

# Note: data is in subdirectories within folders (ie data/train/set_1/img.jpg)
train_dir = 'data/train/'
valid_dir = 'data/validation/'

images_per_val_epoch = len(glob.glob(valid_dir + '/*/*'))
images_per_train_epoch = 9 * images_per_val_epoch

def generator(batch_size, img_dir, training = False, save_to_dir = None):
    """A generator that returns black and white images and color images.

    We keep only last 2 channels in YCrCb space since Y value is obviously same as gray scale."""

    def random_crop(img, ratio):

        w, h = img.shape[0], img.shape[1]
        crop_x1 = int(ratio * random.random() * w)
        crop_y1 = int(ratio * random.random() * h)
        crop_x2 = int((1 - ratio * random.random()) * w)
        crop_y2 = int((1 - ratio * random.random()) * h)

        return img[crop_x1:crop_x2, crop_y1:crop_y2, :]

    bw_images = np.zeros((batch_size, config.width, config.height))
    color_images = np.zeros((batch_size, config.width, config.height, 2))

    while True:

        # Reload list of images (in case we updated it during training)
        image_filenames = glob.glob(img_dir + "/*/*")
        n_files = len(image_filenames)
        random.shuffle(image_filenames)
        for batch_start in range(0, n_files - batch_size + 1, batch_size):
            for i in range(batch_size):

                # Load image
                img_path = image_filenames[batch_start + i]            
                img_BGR = cv2.imread(img_path)

                # Random cropping
                if training and config.crop:
                    img_BGR = random_crop(img_BGR, config.crop)

                # Resizing
                if img_BGR.shape != (256, 256, 3): # it has not been resized yet
                    img_BGR = cv2.resize(img_BGR, (config.width, config.height))
                
                # Random flipping
                if training and random.random() > 0.5:  # we can flip randomly the image
                    img_BGR = cv2.flip(img_BGR, 1)

                # Save data
                if save_to_dir:
                    output_name = Path(img_path).name
                    cv2.imwrite(save_to_dir + '/' + output_name, img_BGR)

                # Convert to YCrCb and yield data as (Y, CrCb)
                img_YCrCb = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
                color_images[i] = img_YCrCb[..., 1:] / 127.5 - 1
                bw_images[i] = img_YCrCb[..., 0] / 127.5 - 1
            yield (bw_images, color_images)

def create_model_and_train(n_layers, n_filters, load_model_path = None):

    if load_model_path:
        model = load_model(load_model_path)

    else:
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

    # Load validation data
    (val_bw_images, val_color_images) = next(generator(images_per_val_epoch, valid_dir))

    # Train
    model.fit_generator(generator(config.batch_size, train_dir, training=True),
                        steps_per_epoch=int(images_per_train_epoch / config.batch_size),
                        epochs=config.num_epochs, callbacks=[WandbCallback(data_type='image', predictions=16),
                        ModelCheckpoint(filepath = 'model/weights.{epoch:03d}.hdf5')],  # TODO add datetime
                        validation_data=(val_bw_images, val_color_images))

if __name__ == "__main__":

    create_model_and_train(config.n_layers, config.n_filters, model_path)