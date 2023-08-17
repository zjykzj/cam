# -*- coding: utf-8 -*-

"""
@date: 2023/8/17 下午8:28
@file: pytorch-grad-cam.py
@author: zj
@description: https://github.com/jacobgil/pytorch-grad-cam
"""

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
import argparse
from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import resnet50


def data_preprocess(bgr_img, target_size=224):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    input_tensor = transform(Image.fromarray(rgb_img))
    return input_tensor


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('img', type=str, help='Image Path.')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()

    # Data
    bgr_img = cv2.imread(opt.img)
    input_tensor = data_preprocess(bgr_img, target_size=224)
    input_tensor = input_tensor.unsqueeze(0)
    # Create an input tensor image for your model..
    # Note: input_tensor can be a batch tensor with several images!

    # Model
    model = resnet50(pretrained=True)
    target_layers = [model.layer4[-1]]

    # Construct the CAM object once, and then re-use it on many images:
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)

    # We have to specify the target we want to generate
    # the Class Activation Maps for.
    # If targets is None, the highest scoring category
    # will be used for every image in the batch.
    # Here we use ClassifierOutputTarget, but you can define your own custom targets
    # That are, for example, combinations of categories, or specific outputs in a non standard model.
    # targets = [ClassifierOutputTarget(281)]
    targets = None
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]
    grayscale_cam = cv2.resize(grayscale_cam, (bgr_img.shape[1], bgr_img.shape[0]))
    visualization = show_cam_on_image(bgr_img.astype(float) / 255, grayscale_cam, use_rgb=False)

    cv2.imshow("bgr_img", bgr_img)
    cv2.imshow("visualization", visualization)
    cv2.waitKey(0)
