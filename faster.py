import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
 
from os import listdir
from os.path import isfile, join
 
import argparse
 
print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))
 
# Parsing command line arguments while making sure they are mandatory/required
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="The directory that contains the input video frames.")
parser.add_argument(
    "--output",
    type=str,
    required=True,
    help="The directory that will contain the output video frames.")
parser.add_argument(
    "--style",
    type=str,
    required=True,
    help="The location of the style frame.")
 
 
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
     
    args = parser.parse_args()
    input_path = args.input + '/'
    output_path = args.output + '/'

    input_path  = "/home/stgeorge/t_money/input/frames/the_cosmic_giggle_360.mp4/"
    output_path = "/array1/home/stgeorge/t_money/output/frames/the_cosmic_giggle_360.mp4/"

    # List all files from the input directory. This directory should contain at least one image/video frame.
    onlyfiles = [f for f in listdir(input_path) if isfile(join(input_path, f))]
 
    # Loading the input style image.
    style_image_path = args.style  # @param {type:"string"}
    style_image_path = "/home/stgeorge/t_money/styles/escher1.jpg"
    style_image = plt.imread(style_image_path)
 
    # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
    style_image = style_image.astype(np.float32)[np.newaxis, ...] / 255.
 
    # Optionally resize the images. It is recommended that the style image is about
    # 256 pixels (this size was used when training the style transfer network).
    # The content image can be any size.
    style_image = tf.image.resize(style_image, (256, 256))
     
    # Load image stylization module.
    # Enable the following line and disable the next two to load the stylization module from a local folder.
    # hub_module = hub.load('magenta_arbitrary-image-stylization-v1-256_2')
    # Disable the above line and enable these two to load the stylization module from the internet.
    hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
    hub_module = hub.load(hub_handle)
  
 
    for inputfile in onlyfiles:
        content_image_path = input_path + inputfile  # @param {type:"string"}
        content_image = plt.imread(content_image_path)
        # Convert to float32 numpy array, add batch dimension, and normalize to range [0, 1]. Example using numpy:
        content_image = content_image.astype(np.float32)[np.newaxis, ...] / 255.
 
        # Stylize image.
        outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
        stylized_image = outputs[0]
 
        # Saving stylized image to disk.
        content_outimage_path = output_path + inputfile  # @param {type:"string"}
        tf.keras.utils.save_img(content_outimage_path, stylized_image[0])