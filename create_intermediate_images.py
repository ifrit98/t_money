#https://www.tensorflow.org/tutorials/generative/style_transfer#define_content_and_style_representations 
import os
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'


import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import PIL.Image
import tensorflow as tf

# # Choose intermediate layers
CONTENT_LAYERS = ['block5_conv2'] 

STYLE_LAYERS = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

NUM_CONTENT_LAYERS = len(CONTENT_LAYERS)
NUM_STYLE_LAYERS = len(STYLE_LAYERS)

DEFAULT_CONTENT_PATH = tf.keras.utils.get_file(
    'YellowLabradorLooking_new.jpg', 
    'https://storage.googleapis.com/download.tensorflow.org/example_images/YellowLabradorLooking_new.jpg')
DEFAULT_STYLE_PATH = tf.keras.utils.get_file(
    'kandinsky5.jpg',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg')


# TODO: setup argparser to catch these variables as defaults
# variables/hparams
save_each_step=True
epochs=50
steps_per_epoch=10
style_dir = "./styles"
content_dir = "./content"
content_nm = "terence" 
content_ext = ".jpg"
results_dir = "./results/{}".format(content_nm)
deltas_img_outpath=os.path.join(results_dir, "deltas")
sobel_img_outpath=os.path.join(results_dir, "sobel")
final_img_outpath=os.path.join(results_dir, "final")
intermediate_imgs_outpath=os.path.join(results_dir, "intermediate")
style_path=os.path.join(style_dir, "tubingen_kandinsky.png")
content_path=os.path.join(content_dir, content_nm + content_ext)
total_variation_weight=30
style_weight=1e-2
content_weight=1e4

# import sys
# import argparse

# def cmdline_args():
#         # Make parser object
#     p = argparse.ArgumentParser(description=__doc__,
#         formatter_class=argparse.RawDescriptionHelpFormatter)
    
#     p.add_argument("required_positional_arg",
#                    help="desc")
#     p.add_argument("required_int", type=int,
#                    help="req number")
#     p.add_argument("--on", action="store_true",
#                    help="include to enable")
#     p.add_argument("-v", "--verbosity", type=int, choices=[0,1,2], default=0,
#                    help="increase output verbosity (default: %(default)s)")
                   
#     group1 = p.add_mutually_exclusive_group(required=True)
#     group1.add_argument('--enable',action="store_true")
#     group1.add_argument('--disable',action="store_false")

#     return(p.parse_args())

# try:
#     args = cmdline_args()
#     print(args)
# except:
#     print('Try $python <script_name> "Hello" 123 --enable')


if not os.path.exists(results_dir):
    os.mkdir(results_dir)

if not os.path.exists(deltas_img_outpath):
    os.mkdir(deltas_img_outpath)

if not os.path.exists(sobel_img_outpath):
    os.mkdir(sobel_img_outpath)

if not os.path.exists(final_img_outpath):
    os.mkdir(final_img_outpath)

if not os.path.exists(intermediate_imgs_outpath):
    os.mkdir(intermediate_imgs_outpath)


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def load_img(path_to_img, max_dim=512):
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

# Calculate style
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

# Build the model
def vgg_layers(layer_names):
  """ Creates a vgg model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False

  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

def style_content_loss(outputs, 
                       style_targets, 
                       content_targets):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / NUM_STYLE_LAYERS

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / NUM_CONTENT_LAYERS
    loss = style_loss + content_loss
    return loss

def clip_0_1(image):
  return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# The regularization loss associated with this is the sum of the squares of the values:
def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))

# Now include it in the train_step function
@tf.function()
def train_step(image, 
               extractor, 
               opt, 
               style_targets, 
               content_targets):
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs, style_targets, content_targets)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

# Total variation loss
def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

# Exctract style and content 
class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


def make_intermediate_style_images(content_path=DEFAULT_CONTENT_PATH, style_path=DEFAULT_STYLE_PATH):

  # content_path = r"C://Users/stgeorge/Desktop/personal_projects/t_money_nft/content/images/tree.jpg"
  content_image = load_img(content_path)
  style_image = load_img(style_path)

  style_extractor = vgg_layers(STYLE_LAYERS)
  style_outputs = style_extractor(style_image*255)

  #Look at the statistics of each layer's output
  print("Statistics for each layer's output:")
  for name, output in zip(STYLE_LAYERS, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()

  print("Calling extractor on content image...")
  extractor = StyleContentModel(STYLE_LAYERS, CONTENT_LAYERS)
  results = extractor(tf.constant(content_image))

  print('Styles:')
  for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

  print("Contents:")
  for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

  # With this style and content extractor, you can now implement the style transfer algorithm. 
  # Do this by calculating the mean square error for your image's output relative to each 
  # target, then take the weighted sum of these losses.
  print("\nStyle transfer algorithm:")
  print("\tCalculate mse for image's output relative to each target and take weighted sum")
  style_targets = extractor(style_image)['style']
  content_targets = extractor(content_image)['content']

  # Reinstantiate image as fresh variable
  image = tf.Variable(content_image)

  # initialize optimizer
  opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

  # Get Deltas and Sobel Edges
  x_deltas, y_deltas = high_pass_x_y(content_image)

  plt.figure(figsize=(14, 10))
  plt.subplot(2, 2, 1)
  imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Original")

  plt.subplot(2, 2, 2)
  imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Original")

  x_deltas, y_deltas = high_pass_x_y(image)

  plt.subplot(2, 2, 3)
  imshow(clip_0_1(2*y_deltas+0.5), "Horizontal Deltas: Styled")

  plt.subplot(2, 2, 4)
  imshow(clip_0_1(2*x_deltas+0.5), "Vertical Deltas: Styled")
  plt.savefig(os.path.join(deltas_img_outpath, content_nm) + content_ext)

  sobel = tf.image.sobel_edges(content_image)

  plt.figure(figsize=(14, 10))
  plt.subplot(1, 2, 1)
  imshow(clip_0_1(sobel[..., 0]/4+0.5), "Horizontal Sobel-edges")
  plt.subplot(1, 2, 2)
  imshow(clip_0_1(sobel[..., 1]/4+0.5), "Vertical Sobel-edges")
  plt.savefig(os.path.join(sobel_img_outpath, content_nm) + content_ext)

  print("Total variation loss: {}".format(total_variation_loss(image).numpy()))

  # Reinstantiate image fresh for training 
  image = tf.Variable(content_image)

  # Run the optimization
  print("Starting optimization...")

  start = time.time()
  step = 0
  for n in range(epochs):
      for m in range(steps_per_epoch):
          step += 1
          train_step(
            image, extractor, opt, style_targets, content_targets)
          print(".", end='', flush=True)

          if save_each_step:
            tensor_to_image(image).save(
              os.path.join(intermediate_imgs_outpath, "{}.png".format(step)))

      tensor_to_image(image).save(
          os.path.join(intermediate_imgs_outpath, "{}.png".format(step)))
      print("Train step: {}".format(step))

  end = time.time()
  print("Total time: {:.1f}".format(end-start))


