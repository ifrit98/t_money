from PIL import Image
import os


path = input("Enter path to gif image:\n> ")
outdir = os.path.join(os.path.dirname(path), os.path.split(path)[-1] + "_frames")
if not os.path.exists(outdir):
    os.mkdir(outdir)

inp = "all" # input("Enter number of key frames to save:\n> ")
if inp == "all":
    with Image.open(path) as im:
        for i in range(im.n_frames):
            im.seek(i)
            im.save(os.path.join(outdir, '{}.png'.format(i)))
else:
    num_key_frames = int(inp)
    with Image.open(path) as im:
        for i in range(num_key_frames):
            im.seek(im.n_frames // num_key_frames * i)
            im.save(os.path.join(outdir, '{}.png'.format(i)))

# C:\Users\stgeorge\Documents\terence\take_it.gif