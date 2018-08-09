# This file processes an image
# If you change the training code, add normalization or something different
# Please make sure this works.
#
# Usage: run.py <model_file_path> <input_image> <output_image>
#
# model_file_path should initially be a path to an h5 file.  By default wandb
# will save a model in wandb/<run_id>/model-best.h5 

import argparse
import keras
from PIL import Image
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Run model file on input_image file and output out_image.')
parser.add_argument('model')
parser.add_argument('input_image')
parser.add_argument('output_image')

height=256
width=256

args = parser.parse_args()
model = keras.models.load_model(args.model)

img = Image.open(args.input_image).resize((width, height))
color_image = np.array(img)

# bw_image is the input image converted to black and white
bw_image = np.array(img.convert('L'))
bw_image = bw_image.reshape((1,width,height))  

recolored_image_array = model.predict(bw_image)

# new_image is the output from the model
new_image = recolored_image_array[0].astype(np.uint8)
new_image = Image.fromarray(new_image, 'RGB' )
new_image.save(args.output_image)
