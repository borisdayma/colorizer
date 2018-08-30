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

# Load model
model = keras.models.load_model(args.model)

# Read and resize image
img_BGR = cv2.imread(args.input_image)
img_BGR = cv2.resize(img_BGR, (width, height))

# Convert to gray
img_gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
img_gray = img_gray.reshape((1,width,height))  

# Preprocess image
img_gray = img_gray / 127.5 - 1

# Predict CrCb channels
img_CrCb = model.predict(img_gray)[0]

# Reconstitute image
img_gray = img_gray[0]
print(img_gray.shape, img_CrCb.shape)
img_YCrCb = np.dstack((img_gray, img_CrCb))
img_YCrCb = (img_YCrCb + 1) * 127.5
img_YCrCb = img_YCrCb.astype(np.uint8)

# Save file
img_colored = cv2.cvtColor(img_YCrCb, cv2.COLOR_YCrCb2BGR)
cv2.imwrite(args.output_image, img_colored)