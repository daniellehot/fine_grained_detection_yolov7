# encoding:utf8
from scipy.io import loadmat
#import pandas as pd
import numpy as np
from PIL import Image
import os
import random
#import matplotlib.pyplot as plt
import cv2 as cv
import shutil



if __name__=="__main__":
    meta = loadmat('cars_annotations/cars_meta.mat')

    labels = list()
    for l in meta['class_names'][0]:
        labels.append(l[0])

    train_folder = "train: /home/daniel/fine_grained_detection_yolov7/standford_cars/images/train/"
    val_folder = "val: /home/daniel/fine_grained_detection_yolov7/standford_cars/images/val/"
    test_folder = "test: /home/daniel/fine_grained_detection_yolov7/standford_cars/images/test/"
    number_of_classes = len(labels)

    print(train_folder)
    print(val_folder)
    print(test_folder)
    print("nc:", number_of_classes)
    print("names:", labels)


