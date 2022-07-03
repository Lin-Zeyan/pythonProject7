# -*- coding:utf-8 -*-
"""
Based on CNN neural  network, training handwritten numeral recognition model model.h5.
The class refers to the trained model for handwritten numeral recognition
"""
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model
from opt_einsum.backends import torch
import torch
import torchvision
import struct
import matplotlib.pyplot as plt
import cv2 as cv
import random
import torch
from torch import nn, optim
import torch.nn.functional as F
from ImageUtils import changeImage28


class Recognition:
    def __init__(self, path):
        #  K.set_image_dim_ordering('th')
        K.image_data_format() == "channels_first"
        self.my_model =load_model(
            'C:/Users/86134/PycharmProjects/pythonProject7/lenet_mnist.h5')

    def readImage2result(self, rimg):
        width, height = rimg.size
        im = rimg.convert("L")#转成灰度图像
        data = im.getdata()
        data = np.matrix(data, dtype='float') / 255.0
        new_data = np.reshape(data * 255.0, (height, width))
        new_data = new_data[np.newaxis,:]
        pred = self.my_model.predict(new_data)
       # classes = np.argmax(pred, axis=1)
       # pred = classes
        return np.argmax([pred[0]])


if __name__ == '__main__':
    re = Recognition()
    rimg = Image.open('only.png')
    rimg = changeImage28(rimg)
    resut = re.readImage2result(rimg)
    print(resut)
