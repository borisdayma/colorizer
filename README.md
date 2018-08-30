# colorizer

This is a fun competition to color images.  

There are two scripts
- color.py training code
- run.py run script

The python script color.py is scaffolding code to automatically load images of flower in black and white and try to learn a network to predict the original color image.  We download some validation data to a directory called test, please don't modify that directory. 

You are welcome to use other ML architectures besides keras, but please use *wandb.log({"loss": loss})* so that we can see your performance.

At the end of the day, we will try your models on new sample images, so please make sure *run.py <model_path> <input_img_path> <output_img_path>* produces nice output and save your model in your wandb directory (our callback will do this autmatically).

Things to try:

- Fancier architectures
- Different loss functions
- Data augmentation
- More training data?

# Instructions

- Remove corrupt files from training dataset as well as low resolution images. They can easily be identified by their size.

# Ideas

- add data from ImageNet

- augment data (crop, flip, scale)