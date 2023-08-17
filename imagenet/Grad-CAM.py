# -*- coding: utf-8 -*-

"""
@date: 2023/8/17 下午2:13
@file: grad-cam.py
@author: zj
@description:

Usage - Specify images and model:
    $ python imagenet/Grad-CAM.py --arch resnet50 dog.jpg

Usage - Viewing CAM for a certain category:
    $ python imagenet/Grad-CAM.py --arch resnet50 --cls-name Norwich_terrier dog.jpg

"""

import os
import cv2
import sys
import argparse

import numpy as np
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet, resnet18, resnet34, resnet50, resnet101, resnet152

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def parse_opt():
    model_names = sorted(name for name in models.resnet.__dict__ if name.islower() and name.startswith('resnet'))
    # print(model_names)

    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='Image Path.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18', choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet18)')
    parser.add_argument('--project', default=ROOT / 'runs/', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--cls-name', type=str, default=None,
                        help="View which category Grad-CAM, default to which one has the highest predicted score.")

    opt = parser.parse_args()
    return opt


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break

        path = Path(p)
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def data_preprocess(bgr_img, target_size=224):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    input_tensor = transform(Image.fromarray(rgb_img))
    return input_tensor


def process(opt):
    img_path, model_arch, save_dir, cls_name = opt.img, opt.arch, opt.save_dir, opt.cls_name

    # Data
    bgr_img = cv2.imread(opt.img)
    input_tensor = data_preprocess(bgr_img, target_size=224)

    # Model
    model = eval(opt.arch)(pretrained=True)
    assert isinstance(model, ResNet)
    classes = np.loadtxt('imagenet/imagenet.names', dtype=str, delimiter=' ').tolist()

    feature_list = []
    grad_list = []

    def backward_hook(module, grad_in, grad_out):
        grad_list.append(grad_out[0].detach().clone().cpu())

    def forward_hook(module, input, output):
        feature_list.append(output.detach().clone().cpu())

    model.layer4[-1].register_forward_hook(forward_hook)
    model.layer4[-1].register_backward_hook(backward_hook)

    # Pred
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    output = model(input_tensor.unsqueeze(0).to(device))

    # Backward
    model.zero_grad()
    if cls_name is not None:
        assert cls_name in classes
        pred_idx = classes.index(cls_name)
    else:
        pred_idx = torch.argmax(output.detach()[0])
        cls_name = classes[pred_idx]
    class_name = classes[pred_idx]
    pred_class = output[0, pred_idx]
    pred_class.backward()

    # Grad-CAM
    grads = grad_list[0].squeeze()
    pooled_grads = torch.nn.functional.adaptive_max_pool2d(grads, (1, 1))
    features = feature_list[0].squeeze()
    for i in range(len(features)):
        features[i] *= pooled_grads[i]

    hmap = torch.mean(features, dim=0)
    hmap = torch.maximum(hmap, torch.zeros(1))
    hmap /= torch.max(hmap)

    # Show
    hmap = np.uint8(hmap * 255)
    print(hmap)
    # cv2.imshow("hmap src", hmap)
    hmap = cv2.resize(hmap, (bgr_img.shape[1], bgr_img.shape[0]))
    hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)

    cv2.imshow("hmap", hmap)
    cv2.imshow("src", bgr_img)

    grad_cam = np.uint8(bgr_img * 0.7 + hmap * 0.3)
    cv2.imshow(class_name, grad_cam)
    cv2.waitKey(0)

    # Save
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    src_path = os.path.join(save_dir, os.path.basename(img_path))
    cv2.imwrite(src_path, bgr_img)

    hmap_path = os.path.join(save_dir, 'hmap.jpg')
    cv2.imwrite(hmap_path, hmap)

    cam_path = os.path.join(save_dir, f'cam_{class_name}.jpg')
    cv2.imwrite(cam_path, grad_cam)

    cmp = np.concatenate([bgr_img, hmap, grad_cam], axis=1)
    cmp_path = os.path.join(save_dir, 'cmp.jpg')
    cv2.imwrite(cmp_path, cmp)


def main(opt):
    opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    print(f"opt: {opt}")

    process(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
