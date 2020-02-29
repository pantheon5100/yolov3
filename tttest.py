import argparse
# from ConfigParser import ConfigParser
import shlex
from utils.parse_config import *

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
parser.add_argument('--accumulate', type=int, default=4, help='batches to accumulate before optimizing')
parser.add_argument('--cfg', type=str, default='cfg/yolov3-tiny.cfg', help='*.cfg path')
parser.add_argument('--data', type=str, default='data/hg.data', help='*.data path')
parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')
parser.add_argument('--img-size', nargs='+', type=int, default=[416], help='train and test image-sizes')
parser.add_argument('--rect', action='store_true', help='rectangular training')
parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
parser.add_argument('--notest', action='store_true', help='only test final epoch')
parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters', default=True)
parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
parser.add_argument('--weights', type=str, default='weights/yolov3-tiny.conv.15', help='initial weights path')
parser.add_argument('--arc', type=str, default='default', help='yolo architecture')  # default, uCE, uBCE
parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
parser.add_argument('--var', type=float, help='debug variable')
# opt = parser.parse_args()

# load local args
a = parse_args_cfg('zk_train0228.args')
print(a)
# print(parser.parse_args())
opt = parser.parse_args(a)
print(opt)
