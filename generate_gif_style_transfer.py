import os
import tensorflow as tf
# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import IPython.display as display

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image
import time
import functools

import imageio
import glob

import argparse

import moviepy.editor as mp
import ffmpy

import tensorflow_hub as hub

nss = numeric_str_sort = lambda l: l.sort(key=lambda x: int(x.split("\\")[-1].split(".")[0]))

def load_model(hub_path='https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'):
    hub_model = hub.load(hub_path)
    return hub_model


HUB_MODEL = load_model()


def get_online_images(content_url, style_url):
    content_path = tf.keras.utils.get_file(os.path.split(content_url)[-1], content_url)
    style_path = tf.keras.utils.get_file(os.path.split(style_url)[-1], style_path)
    return content_path, style_path


def get_frame_filepaths(basepath, ext='.png'):
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

def extract_frames(gifpath, outdir):
  with PIL.Image.open(gifpath) as im:
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
    plt.show()


def stylize_image(content_path, style_path, show=False):
    if globals().get("HUB_MODEL") is None:
        globals().update({"HUB_MODEL", load_model()})

    content_image = load_img(content_path)
    style_image = load_img(style_path)

    if show:
        plt.subplot(1, 2, 1)
        imshow(content_image, 'Content Image')

        plt.subplot(1, 2, 2)
        imshow(style_image, 'Style Image')

    stylized_image = HUB_MODEL(tf.constant(content_image), tf.constant(style_image))[0]
    if show:
        imshow(stylized_image)

    return tensor_to_image(stylized_image)

VIDEO_PATH = "./videos/"
GIF_PATH = "./gifs/"

def convert_gif_to_video(gif_path):
  # # A
  # import ffmpy
  # ff = ffmpy.FFmpeg(
  #   inputs={gif_path: None}, outputs={gif_path.split(".gif")[0] + '.mp4': None})
  # B
  clip = mp.VideoFileClip(gif_path)
  clip.write_videofile(gif_path.split(".gif")[0] + '.mp4')

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def stylize_gif(gif_frames_dirpath, style_path):
    files = get_frame_filepaths(gif_frames_dirpath)
    for fp in files:
        op = fp.split(".png")[0] + "_styled.png"
        styled_image = stylize_image(fp, style_path)
        styled_image.save(op)



if False:
  path = "/home/user/internal/t_money/gifs/take_it_frames"
  path = "./take_it_frames"
  style_path = "../styles/Edvard-Munch.jpg"
  gif_path = "take_it.gif"


# parser = argparse.ArgumentParser()
# parser.add_argument('--content_path', type=dir_path)
# parser.add_argument('--style_path', type=dir_path)