from create_intermediate_images import make_intermediate_style_images
cp = ""
sp = ""
outpath = ""

# run create_intermediate_images script
make_intermediate_style_images(cp, sp, outpath)

# turn into a GIF.
from utils.utils import frames_to_gif
gif_outpath = ".gif"
frames_to_gif(outpath, gif_outpath)