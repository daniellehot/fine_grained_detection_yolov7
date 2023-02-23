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

def create_class_to_id_map(_labels):
    class_to_id_map = {}
    for label in _labels:
        class_to_id_map[label] = _labels.index(label)
    id_to_class_map = dict(zip(class_to_id_map.values(), class_to_id_map.keys()))
    return class_to_id_map, id_to_class_map
     

def convert_to_YOLO_format(_annotations, _class_to_id_map, _img_path):
    converted_annotations = list()
    for entry in _annotations:
        class_id = _class_to_id_map[entry[0]]
        image_file = os.path.join(_img_path, entry[1])
        image_w, image_h = (Image.open(image_file)).size
        x_min, y_min, x_max, y_max = float(entry[2]), float(entry[3]), float(entry[4]), float(entry[5])
        b_center_x = (x_min + x_max)/2 
        b_center_y = (y_min + y_max)/2
        b_width    = x_max - x_min
        b_height   = y_max - y_min
        b_center_x /= image_w 
        b_center_y /= image_h 
        b_width    /= image_w 
        b_height   /= image_h 
        converted_annotations.append((class_id, b_center_x, b_center_y, b_width, b_height, entry[1]))
    return converted_annotations


def visualise_random_image_pair(_annotations, _converted_annotations, _img_path, _id_to_class_map):
    rnd_idx = random.randrange(len(_annotations))
    annotation = _annotations[rnd_idx]
    annotation_c =_converted_annotations[rnd_idx] 

    image_file = os.path.join(_img_path, annotation[1])

    #Original annotations
    image = cv.imread(image_file)
    x_min, y_min, x_max, y_max = annotation[2], annotation[3], annotation[4], annotation[5]
    cv.rectangle(image, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    cv.putText(image, annotation[0], (x_min, y_min-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    #Converted annotations
    image_c = cv.imread(image_file)
    height = image.shape[0]
    width = image.shape[1]
    label = _id_to_class_map[annotation_c[0]]
    center_x = annotation_c[1]*width
    center_y = annotation_c[2]*height
    width = (annotation_c[3]*width)/2
    height = (annotation_c[4]*height)/2
    x_min = int(center_x - width)
    x_max = int(center_x + width)
    y_min = int(center_y - height)
    y_max = int(center_y + height)
    cv.rectangle(image_c, (x_min, y_min), (x_max, y_max), (0,0,255), 2)
    cv.putText(image_c, label, (x_min, y_min-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    images_stacked = np.concatenate((image, image_c), axis=1)
    cv.imshow("compare", images_stacked)
    cv.waitKey()

def move_files(_files, _current_destination, _goal_img_destination, _goal_annotations_destination):
    for entry in _files:
        filename = entry[-1]
        current_path = os.path.join(_current_destination, filename)
        goal_img_path = os.path.join(_goal_img_destination, filename)
        filename = filename.replace(".jpg",".txt")
        goal_annotation_path = os.path.join(_goal_annotations_destination, filename)
        print(current_path)
        print(goal_img_path)
        print(goal_annotation_path)
        #print(entry[:-1])
        string_to_save = str(entry[0]) + " " + str(entry[1]) + " " + str(entry[2]) + " " + str(entry[3])  + " " + str(entry[4])
        with open(goal_annotation_path, 'w') as f:
            f.write(string_to_save)
        exit(50)
        

if __name__=="__main__":
    mat_train = loadmat('cars_annotations/cars_train_annos.mat')
    mat_test = loadmat('cars_annotations/cars_test_annos_withlabels.mat')
    meta = loadmat('cars_annotations/cars_meta.mat')

    labels = list()
    for l in meta['class_names'][0]:
        labels.append(l[0])

    class_to_id_map, id_to_class_map = create_class_to_id_map(labels)

    train = list()
    for example in mat_train['annotations'][0]:
        label = labels[example[-2][0][0]-1]
        image = example[-1][0]
        x_min = example[0][0][0]
        y_min = example[1][0][0]
        x_max = example[2][0][0]
        y_max = example[3][0][0]
        train.append((label, image, x_min, y_min, x_max, y_max))

    test = list()
    for example in mat_test['annotations'][0]:
        #image = example[-1][0]
        #test.append(image)
        label = labels[example[-2][0][0]-1]
        image = example[-1][0]
        x_min = example[0][0][0]
        y_min = example[1][0][0]
        x_max = example[2][0][0]
        y_max = example[3][0][0]
        test.append((label, image, x_min, y_min, x_max, y_max))

    #print("Train before split", len(train))
    validation_size = int(len(train) * 0.10)
    validation = train[:validation_size].copy()
    #np.random.shuffle(validation)
    train = train[validation_size:]

    #print("Train after split", len(train))
    #print("Validation", len(validation))
    #print("Test", len(test))
    #test_size = int(len(train) * 0.20)
    #test = train[:test_size].copy()
    #np.random.shuffle(test)
    #train = train[test_size:]

    train_converted = convert_to_YOLO_format(train, class_to_id_map, "cars_train")
    #print("train converted", len(train_converted))
    validation_converted = convert_to_YOLO_format(validation, class_to_id_map, "cars_train")
    test_converted = convert_to_YOLO_format(test, class_to_id_map, "cars_test")

    #visualise_random_image_pair(train, train_converted, "cars_train", class_to_id_map, id_to_class_map)
    visualise_random_image_pair(_annotations = train,
        _converted_annotations = train_converted,
        _img_path = "cars_train", 
        _id_to_class_map = id_to_class_map)

    move_files(_files = train_converted, _current_destination = "cars_train", _goal_img_destination = "images/train", _goal_annotations_destination = "annotations/train")
    move_files(_files = validation_converted, _current_destination = "cars_train", _goal_img_destination = "images/val", _goal_annotations_destination = "annotations/val")
    move_files(_files = test_converted, _current_destination = "cars_test", _goal_img_destination = "images/test", _goal_annotations_destination = "annotations/test")
    
    """
    train_path = 'car_devkit/train/cars_train/'
    test_path = 'car_devkit/test/cars_test/'

    with open('car_devkit/cars_data.csv', 'w+') as f:
        [f.write('TRAIN,%s%s,%s,%s,%s,%s,%s\n' %(train_path, img, bbox_x1, bbox_x2, bbox_y1, bbox_y2, lab)) for img, bbox_x1, bbox_x2, bbox_y1, bbox_y2, lab in train]
        [f.write('VALIDATION,%s%s,%s,%s,%s,%s,%s\n' %(train_path, img, bbox_x1, bbox_x2, bbox_y1, bbox_y2, lab)) for img, bbox_x1, bbox_x2, bbox_y1, bbox_y2, lab in validation]
        [f.write('TEST,%s%s\n' %(test_path,img)) for img,_,_,_,_,_ in test]# encoding:utf8
    """