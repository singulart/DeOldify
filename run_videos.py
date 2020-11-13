from deoldify import device
from deoldify.device_id import DeviceId
#choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
import sys
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

colorizer = get_video_colorizer()

source_url = sys.argv[1] #@param {type:"string"}
print(source_url)
render_factor = 35  #@param {type: "slider", min: 7, max: 40}
watermarked = False #@param {type:"boolean"}

if source_url is not None and source_url !='':
	video_path = colorizer.colorize_from_url(source_url, sys.argv[1], render_factor, watermarked=watermarked)
else:
    print('Provide an video url and try again.')