# -*- coding: utf-8 -*-

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization, \
    GlobalAveragePooling2D

def vgg_block(num_convs, num_channels):
    blk = Sequential()
    for _ in range(num_convs):
        blk.add(Conv2D(num_channels, kernel_size=3,
                       padding='Same', activation='relu'))
    blk.add(MaxPool2D(pool_size=2, strides=1))
    return blk


def vgg(conv_arch):
    net = Sequential()
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    net.add(Dense(256, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Dense(256, activation='relu'))
    net.add(Dropout(0.5))
    net.add(Flatten())
    net.add(Dense(10, activation='softmax'))
    return net


conv_arch = [(1,64),(2,125)]
net = vgg(conv_arch)
