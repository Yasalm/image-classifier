import torch


from utils import load_checkpoint, predict
import argparse
import pandas as pd


parser = argparse.ArgumentParser(description="Parser of predicitng script")

parser.add_argument(
    '--load_dir', help='Provide loadinig directory for pre-model.  Optional argument ', type=str)
parser.add_argument('--gpu', help="Option to use GPU", type=str)
parser.add_argument('--topk', help="Option to use Top K classes", type=int)
parser.add_argument('--image_path', help="Path of image", type=str)

## parser.add_argument ('--modelname', help = "Model name -- default = densenet121, options -- densenet121 - resnet50 ", type = str)

args = parser.parse_args()

if args.load_dir:
    load_dir = args.load_dir
else:
    load_dir = 'checkpoint.pth'

if args.gpu:
    device = 'cuda'
else:
    device = 'cpu'
if args.topk:
    topk = args.topk
else:
    topk = 5

if args.image_path:
    img_path = args.image_path
else:
    img_path = 'flowers/test/100/image_07896.jpg'


if __name__ == "__main__":
    prob, _, flowers = predict(load_dir, img_path, topk, device)
    # to display as precentage.
    prob = ["{0:.2f}%".format(val * 100) for val in prob]

data = {
    'Flowers': flowers,
    'Probability': prob,
}

df = pd.DataFrame(data)
print(df)
