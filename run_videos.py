from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import *
import warnings

# choices:  CPU, GPU0...GPU7
device.set(device=DeviceId.GPU0)
import torch

if not torch.cuda.is_available():
    print('GPU not available.')

torch.backends.cudnn.benchmark = True

warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")

render_factor = int(sys.argv[2]) if sys.argv[2] else 7

colorizer = get_video_colorizer(render_factor=render_factor)

source_url = sys.argv[1]
print(source_url)
watermarked = False

if source_url is not None and source_url != '':
    video_path = colorizer.colorize_from_url(source_url, sys.argv[1], render_factor, watermarked=watermarked)
else:
    print('Provide an video url and try again.')
