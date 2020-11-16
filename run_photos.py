from deoldify import device
from deoldify.device_id import DeviceId
import glob
device.set(device=DeviceId.GPU0)
import torch

if not torch.cuda.is_available():
    print('GPU not available.')

from os import path
import fastai
from deoldify.visualize import *
from pathlib import Path
torch.backends.cudnn.benchmark=True
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

colorizer = get_image_colorizer(artistic=True)

source_folder = sys.argv[1]
render_factor = 35
watermarked = False

if source_folder is not None and source_folder != '':
    src_path = Path(source_folder)

    for g in src_path.glob('**/*'):
        for rf in range(8, 41, 2):
            image_path = colorizer.plot_transformed_image(str(g), render_factor=rf, compare=True, watermarked=watermarked)
            suffix = image_path.suffix
            stem = image_path.stem
            parent = g.parent
            new_name = "".join([str(stem), '_', str(rf) + suffix])

            new_place = Path(parent, new_name)
            image_path.rename(new_place)
            print('Saved to %s' % str(new_place))
else:
    print('Usage: python run_photos.py /path/to/folder/with/bw/images')
