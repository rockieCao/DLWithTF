# -*- coding: utf-8 -*-

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization, \
    GlobalAveragePooling2D


def nin_block(num_channels, kernel_size, strides, padding='Valid'):
    blk = Sequential()
    blk.add(Conv2D(filters=num_channels, kernel_size=kernel_size, strides=strides, padding=padding, activation='relu'))
    blk.add(Conv2D(filters=num_channels, kernel_size=1, activation='relu'))
    blk.add(Conv2D(filters=num_channels, kernel_size=1, activation='relu'))
    return blk


net = Sequential()
net.add(nin_block(96, kernel_size=7, strides=2))
net.add(MaxPool2D(pool_size=2, strides=2))
net.add(nin_block(256, kernel_size=5, strides=1, padding='Same'))
net.add(MaxPool2D(pool_size=2, strides=2))
net.add(nin_block(384, kernel_size=3, strides=1, padding='Same'))
net.add(MaxPool2D(pool_size=2, strides=2))
net.add(Dropout(0.5))
net.add(nin_block(10, kernel_size=3, strides=1, padding='Same'))
net.add(GlobalAveragePooling2D())
# net.add(Flatten())
