import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

import matplotlib.pyplot as plt

DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()
    plt.imshow(img_flo[:, :, [2,1,0]]/255.0, cmap='gray')
    plt.show()


def demo(args, reverse=False):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        if reverse:
            images.reverse()

        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            # viz(image1, flow_up)
            flo = flow_up[0].permute(1,2,0).cpu().numpy()

            if reverse:
                np.save(os.path.join(args.out_path, os.path.basename(imfile2).replace('.jpg', '.npy')), flo)
            else:
                np.save(os.path.join(args.out_path, os.path.basename(imfile1).replace('.jpg', '.npy')), flo)


to_do = ['bear', 'parkour', 'breakdance-flare', 'motocross-bumps', 'scooter-gray', 'train', 'libby', 'horsejump-high', 'soapbox', 'dance-twirl']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--out_path')
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    base_path = args.path
    image_folders = os.listdir(os.path.join(base_path, 'JPEGImages', '480p'))
    for folder in image_folders:

        if folder not in to_do:
            continue

        print(folder)
        args.path = os.path.join(base_path, 'JPEGImages', '480p', folder)
        args.out_path = os.path.join(base_path, 'FwFlow', folder)
        os.makedirs(args.out_path, exist_ok=True)

        demo(args, reverse=False)

        args.path = os.path.join(base_path, 'JPEGImages', '480p', folder)
        args.out_path = os.path.join(base_path, 'BwFlow', folder)
        os.makedirs(args.out_path, exist_ok=True)

        demo(args, reverse=True)

