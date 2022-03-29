import functools
import os

from matplotlib import gridspec
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import PIL

print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

RESULTS_DIR = "./results"
CROP = False
SHOW = False

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def save_tensor_as_image(tensor, outpath):
  im = tensor_to_image(tensor)
  im.save(outpath)

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=None)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = tf.io.decode_image(
      tf.io.read_file(image_path),
      channels=3, dtype=tf.float32)[tf.newaxis, ...]
  img = crop_center(img) if CROP else img
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=preserve_aspect_ratio)
  return img

def show_n(images, titles=('',)):
  n = len(images)
  image_sizes = [image.shape[1] for image in images]
  w = (image_sizes[0] * 6) // 320
  plt.figure(figsize=(w * n, w))
  gs = gridspec.GridSpec(1, n, width_ratios=image_sizes)
  for i in range(n):
    plt.subplot(gs[i])
    plt.imshow(images[i][0], aspect='equal')
    plt.axis('off')
    plt.title(titles[i] if len(titles) > i else '')
  plt.show()



DEFAULT_CONTENT_URLS = dict(
  terence0='https://i.pinimg.com/236x/40/41/3c/40413c9252dc72a3d48eed8a5d37303b--terence-mckenna-psychedelic.jpg',
  terence1='https://pbs.twimg.com/profile_images/846009016467836928/JnT82hu0_400x400.jpg',
  terence2='https://lucys-magazin.com/wp-content/uploads/Terence-McKenna.jpg',
  terence3='https://www.knihydobrovsky.cz/thumbs/book-preview-big/mod_eshop/produkty/t/trialogy-na-hranicich-zapadu-9788086685090.jpg',
  terence4='https://images.squarespace-cdn.com/content/v1/60de73452049513d11bedcf8/1632247344431-3GHA0Q15F3BEFEFHGFVU/Mckenna008.jpg',
  terence5='https://i1.sndcdn.com/artworks-Wosvz9N8gxgzwvaT-IlFEzw-t500x500.jpg',
  terence6='https://1.bp.blogspot.com/-NJpaB8Jvta4/WwM3kSmUHSI/AAAAAAAArFI/HlrMN0Fhgb4VS2iMz7ZmZsHx-SZ3vJsewCLcBGAs/s1600/fullsizeoutput_4ee3.jpeg',
  terence7='https://doorofperception.com/wp-content/uploads/doorofperception.com-terence_mckenna.jpg',
  terence8='https://images.squarespace-cdn.com/content/v1/56d3031b746fb93b0ba4ab8b/1551710327368-8MFBJFWJIQ7SK25MRNG1/Screenshot+2019-03-04+at+15.38.33.png',
  terence9='https://i.pinimg.com/564x/06/85/99/068599a51c2d9c9d5446ad83fcb3bc7b.jpg',
  terence10='https://edgarperacinema.files.wordpress.com/2020/04/hi8-hc-208-barco-terence-ri10.jpg',
  # terence_dennis='https://miro.medium.com/max/1260/1*3dyfnqaTulzuqNkoYJTcOQ.jpeg'
  )

DEFAULT_STYLE_URLS = dict(
  kanagawa_great_wave='https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Great_Wave_off_Kanagawa.jpg',
  kandinsky_composition_7='https://upload.wikimedia.org/wikipedia/commons/b/b4/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg',
  hubble_pillars_of_creation='https://upload.wikimedia.org/wikipedia/commons/6/68/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg',
  van_gogh_starry_night='https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/1024px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg',
  turner_nantes='https://upload.wikimedia.org/wikipedia/commons/b/b7/JMW_Turner_-_Nantes_from_the_Ile_Feydeau.jpg',
  munch_scream='https://upload.wikimedia.org/wikipedia/commons/c/c5/Edvard_Munch%2C_1893%2C_The_Scream%2C_oil%2C_tempera_and_pastel_on_cardboard%2C_91_x_73_cm%2C_National_Gallery_of_Norway.jpg',
  picasso_demoiselles_avignon='https://upload.wikimedia.org/wikipedia/en/4/4c/Les_Demoiselles_d%27Avignon.jpg',
  picasso_violin='https://upload.wikimedia.org/wikipedia/en/3/3c/Pablo_Picasso%2C_1911-12%2C_Violon_%28Violin%29%2C_oil_on_canvas%2C_Kr%C3%B6ller-M%C3%BCller_Museum%2C_Otterlo%2C_Netherlands.jpg',
  picasso_bottle_of_rum='https://upload.wikimedia.org/wikipedia/en/7/7f/Pablo_Picasso%2C_1911%2C_Still_Life_with_a_Bottle_of_Rum%2C_oil_on_canvas%2C_61.3_x_50.5_cm%2C_Metropolitan_Museum_of_Art%2C_New_York.jpg',
  fire='https://upload.wikimedia.org/wikipedia/commons/3/36/Large_bonfire.jpg',
  derkovits_woman_head='https://upload.wikimedia.org/wikipedia/commons/0/0d/Derkovits_Gyula_Woman_head_1922.jpg',
  amadeo_style_life='https://upload.wikimedia.org/wikipedia/commons/8/8e/Untitled_%28Still_life%29_%281913%29_-_Amadeo_Souza-Cardoso_%281887-1918%29_%2817385824283%29.jpg',
  derkovtis_talig='https://upload.wikimedia.org/wikipedia/commons/3/37/Derkovits_Gyula_Talig%C3%A1s_1920.jpg',
  amadeo_cardoso='https://upload.wikimedia.org/wikipedia/commons/7/7d/Amadeo_de_Souza-Cardoso%2C_1915_-_Landscape_with_black_figure.jpg'
)


# Visualize input images and the generated stylized image.
# show_n([content_image, style_image, stylized_image], titles=['Original content image', 'Style image', 'Stylized image'])
def fast_stylize_url(content_urls=None, style_urls=None):
  """
  Stylize and save cartesian product of input content images cross style images from URLs.

  Param:
    content_urls: dict (k: str name, v: str url)
    style_urls: dict (k: str name, v: str url)

  Returns:
    None
  """

  if not content_urls and not style_urls:
    content_urls = DEFAULT_CONTENT_URLS
    style_urls = DEFAULT_STYLE_URLS

  # # Load TF Hub module.
  hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
  hub_module = hub.load(hub_handle)

  content_image_size = 384
  style_image_size = 256
  content_images = {k: load_image(v, (content_image_size, content_image_size)) for k, v in content_urls.items()}
  style_images = {k: load_image(v, (style_image_size, style_image_size)) for k, v in style_urls.items()}
  style_images = {k: tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME') for k, style_image in style_images.items()}

  stylized_content = {k: {_: None for _ in style_images.keys()} for k in content_images.keys()}


  from itertools import product
  lists = [content_images.keys(), style_images.keys()]
  for content_name, style_name in product(*lists):

      stylized_image = hub_module(tf.constant(content_images[content_name]),
                                  tf.constant(style_images[style_name]))[0]

      stylized_content[content_name][style_name] = stylized_image

      outpath = os.path.join(RESULTS_DIR, "{}_{}.png".format(content_name, style_name))
      save_tensor_as_image(stylized_image, outpath)
      print("saved to {}".format(outpath))
      if SHOW:
        show_n([content_images[content_name], style_images[style_name], stylized_image],
            titles=['Original content image', 'Style image', 'Stylized image'])

