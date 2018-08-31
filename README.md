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

# Prepare data (optional)

- Remove corrupt files from training dataset as well as low resolution images. They can easily be identified by their size. Use the command `find . -name "*" -size -16k`

- Use `resize_and_save_data` to preprocess data and `split_train_valid_data` to split it into train and validation datasets

# Instructions

- Download prepared datasets with `wget`

  - https://www.dropbox.com/s/gxbltbsgg3sgb5t/data1.tar.gz?dl=1
  

# Ideas

- preprocess all images before training and save to disk as numpy arrays or resized images

- add data from ImageNet

- upload standard training/validation/test data (preprocessed) to bucket

- create a larger validation set only from provided images

- augment data (crop, flip, scale)

- use ResNet (pre-trained or not) for first layers (at least up to block 4)

- keep list of accuracy for hard mining and do weighted sample based on accuracy at last epoch

- use weight decay (0.001 as a start) if overfitting

- update learning rate after a certain number of epochs

- perform inference from ensemble of models (either on full picture or different crops/scales)
