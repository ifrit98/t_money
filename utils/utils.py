import PIL
import os
import glob
import numpy as np
import tensorflow as tf
# import tensorflow_hub as hub
import imageio
import matplotlib.pyplot as plt
import moviepy.editor as mp

from tensorflow.python.framework.ops import Tensor, EagerTensor

is_tensor = lambda x: True if x.__class__ in [Tensor, EagerTensor] else False
as_tensor = lambda x, dtype: tf.cast(x, dtype=dtype)

# Expects filepaths be of the form: [dir/0.png, dir/1.png, ... dir/n.png]
nss = numeric_str_sort = lambda l: l.sort(
  key=lambda x: int(x.split("\\")[-1].split(".")[0])
)
tss = terence_str_sort = lambda l: l.sort(
  key=lambda x: int(x.split("\\")[-1].split("_")[-1].split(".")[0])
)
sss = lambda l: l.sort(
  key=lambda x: int(x.split("\\")[-1].split(".")[0].split("_")[0])
)

dirpath=r"C:\Users\stgeorge\Desktop\personal_projects\t_money_nft\content\gifs\take_it_frames\styled"
ext=".png"
outpath=r"C:\Users\stgeorge\Desktop\personal_projects\t_money_nft\results\terence\final\styled_gif.gif"

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_image_dir(dirpath, ext='.png', sort_fn=numeric_str_sort, first=None):
  files = glob.glob(os.path.join(dirpath, "*{}".format(ext)))
  sort_fn(files)
  if first is not None:
    files = files[:int(first)]
  images = []
  for fp in files:
    images.append(load_img(fp))
  return images

def save_tensors_as_gif(images, outpath):
    image_tensors = images if is_tensor(images[0]) \
                           else [as_tensor(img, img.dtype) for img in image_tensors]
    image_objects = [tensor_to_image(img) for img in image_tensors]
    initial = image_objects[0]
    append_images = image_objects[1:] + image_objects[::-1]
    initial.save(
        outpath, save_all=True, append_images=append_images, loop=0
    )
    del images
    print("Completed!")

def convert_gif_to_video(gif_path):
  clip = mp.VideoFileClip(gif_path)
  clip.write_videofile(gif_path.split(".gif")[0] + '.mp4')

# WIP
def frames_to_video(path_input_frames, path_output_video):

    image_list = []
    count = 0
    path_converted_frame = PATH_TMP + 'x' + (str(count).zfill(5)) + '.jpg'

    image = cv2.imread(path_converted_frame)
    height, width, _ = image.shape
    size = (width,height)
    print('size: ', size)

    converted_files = [file_name for file_name in os.listdir(PATH_TMP) if 'x' in file_name]
    converted_files.sort()

    for file_name in converted_files:

        path_converted_frame = PATH_TMP + file_name
        image = cv2.imread(path_converted_frame)
        print(path_converted_frame)
        image_list.append(image)

    video_writer = cv2.VideoWriter(path_output_video, cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, size)
    for i in range(len(image_list)):
        video_writer.write(image_list[i])

    video_writer.release()
    print('video generated: ', path_output_video)

import cv2

def video_to_frames(path_input_video, max_frames=1000, path_tmp='./tmp/'): 
    video_capture = cv2.VideoCapture(path_input_video) 
    if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)
    for count in range(max_frames):
        success, image = video_capture.read() 
        if success == False: break
        path_frame = path_tmp + (str(count).zfill(5)) + '.jpg'
        cv2.imwrite(path_frame, image) 
        print(count, path_frame)

def frames_to_gif(inpath, outpath, sort_fn=numeric_str_sort, ext=".png"):
    images = load_image_dir(inpath, sort_fn=sort_fn, ext=ext)
    save_tensors_as_gif(images, outpath)
    
def gif_to_frames(path, n_frames="all"):
  outdir = os.path.join(os.path.dirname(path), os.path.split(path)[-1] + "_frames")
  if not os.path.exists(outdir):
      os.mkdir(outdir)

  if n_frames == "all":
      with PIL.Image.open(path) as im:
          for i in range(im.n_frames):
              im.seek(i)
              im.save(os.path.join(outdir, '{}.png'.format(i)))
  else:
      num_key_frames = int(n_frames)
      with PIL.Image.open(path) as im:
          for i in range(num_key_frames):
              im.seek(im.n_frames // num_key_frames * i)
              im.save(os.path.join(outdir, '{}.png'.format(i)))

split_gif=gif_to_frames

def load_model(hub_path='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'):
    hub_model = hub.load(hub_path)
    return hub_model

def get_online_images(content_url, style_url):
    content_path = tf.keras.utils.get_file(os.path.split(content_url)[-1], content_url)
    style_path = tf.keras.utils.get_file(os.path.split(style_url)[-1], style_path)
    return content_path, style_path

def get_frame_filepaths(basepath, ext=".png"):
    files = glob.glob(os.path.join(basepath, "*{}".format(ext)))
    numeric_str_sort(files)
    return files

def load_frames(path):
    frames = []
    files = get_frame_filepaths(path)
    for im_path in files:
        im = imageio.imread(im_path)
        frames.append(im)
    return np.stack(frames)

def extract_frames(path, outdir):
  with PIL.Image.open(path) as im:
      for i in range(im.n_frames):
            im.seek(i)
            im.save(os.path.join(outdir, '{}.png'.format(i)))

def tensor_to_image(tensor):
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def load_img(path_to_img):
    max_dim = 512
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