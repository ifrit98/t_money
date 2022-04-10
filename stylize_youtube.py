import os
import argparse
import tensorflow as tf
import tensorflow_hub as hub

# Parse arguments 
print("TF Version: ", tf.__version__)
print("TF Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))
 
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


# Parsing command line arguments while making sure they are mandatory/required
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_url",
    type=str,
    default="https://www.youtube.com/watch?v=FXtWoFLG-p4", # Terence on JRE
    # default="https://www.youtube.com/watch?v=OnIM9gBEZiA",
    # required=True,
    help="URL to YouTube video.")
parser.add_argument(
    "--output_dir",
    type=str,
    default="./results",
    # required=True,
    help="The directory that will contain the output stylized video.")
parser.add_argument(
    "--style_url",
    type=str,
    default=DEFAULT_STYLE_URLS['kanagawa_great_wave'],
    # required=True,
    help="The location (URL) of the style frame.")
    
args = parser.parse_args()
print(vars(args))

# Download youtube url
from utils.utils import download_youtube
yt_path = download_youtube(args.input_url, res='lo')

# Download style image url
style_path = tf.keras.utils.get_file(os.path.split(args.style_url)[-1], args.style_url)


# Split into frames
from utils.utils import generate_frames, extract_mp3, generate_video, add_mp3

import time
time_start = time.time()

# load and cache the styling dnn
HUB_URL = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(HUB_URL)

# extract audio from the video
extract_mp3(yt_path)

# extract all frames from the video, style them, and put results into tmp
generate_frames(yt_path, style_path)

# regenerate the video from the styled frames
name_style = os.path.split(style_path)[-1]
name_original = os.path.split(yt_path)[-1]

output_name = os.path.splitext(name_original)[0] + '.' + os.path.splitext(name_style)[0] + '.mp4'
output_path = os.path.join(args.output_dir, output_name)
generate_video(output_path)

# recombine the extracted audio into the newly-styled video
input_name = output_name
output_name = os.path.splitext(name_original)[0] + '.' + os.path.splitext(name_style)[0] + '.audio.mp4'
add_mp3(os.path.join(args.output_dir, input_name), os.path.join(args.output_dir, output_name))

time_end = time.time()
elapsed_time = int(time_end - time_start)

minutes = int(elapsed_time / 60)
seconds = int(elapsed_time % 60)
print(f'completed: {os.path.join(args.output_dir, output_name)}')
print(f'elapsed time: {minutes} minutes {seconds} seconds')
print(f'elapsed time: {elapsed_time}')

