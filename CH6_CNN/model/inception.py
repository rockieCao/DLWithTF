# -*- coding: utf-8 -*-

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, ZeroPadding2D, Input


def inception_blk(inpt, c1, c2, c3, c4):
    p1 = Conv2D(filters=c1, kernel_size=1, activation='relu')(inpt)

    p2 = Conv2D(filters=c2[0], kernel_size=1, activation='relu')(inpt)
    p2 = ZeroPadding2D(padding=(1,1))(p2)
    p2 = Conv2D(filters=c2[1], kernel_size=3, activation='relu')(p2)

    p3 = Conv2D(filters=c3[0], kernel_size=1, activation='relu')(inpt)
    p3 = ZeroPadding2D(padding=(2,2))(p3)
    p3 = Conv2D(filters=c3[1], kernel_size=5, activation='relu')(p3)

    p4 = ZeroPadding2D(padding=(1,1))(inpt)
    p4 = MaxPool2D(pool_size=3, strides=1)(p4)
    p4 = Conv2D(filters=c4, kernel_size=1, activation='relu')(p4)
    return keras.layers.concatenate([p1, p2, p3, p4], axis=3)


def build_net(input_shape):
    inpt = Input(input_shape)
    
    b1 = ZeroPadding2D(padding=(3,3))(inpt)
    b1 = Conv2D(64, kernel_size=7, strides=2, activation='relu')(b1)
    b1 = ZeroPadding2D(padding=(1,1))(b1)
    b1 = MaxPool2D(pool_size=3, strides=2)(b1)

    b2 = Conv2D(64, kernel_size=1, activation='relu')(b1)
    b2 = ZeroPadding2D(padding=(1,1))(b2)
    b2 = Conv2D(192, kernel_size=3, activation='relu')(b2)
    b2 = ZeroPadding2D(padding=(1,1))(b2)
    b2 = MaxPool2D(pool_size=3, strides=2)(b2)

    b3 = inception_blk(b2, 64, (96, 128), (16, 32), 32)
    b3 = inception_blk(b3, 128, (128, 192), (32, 96), 64)
    b3 = ZeroPadding2D(padding=(1,1))(b3)
    b3 = MaxPool2D(pool_size=3, strides=2)(b3)

    b4 = inception_blk(b3, 192, (96, 208), (16, 48), 64)
    b4 = inception_blk(b4, 160, (112, 224), (24, 64), 64)
    b4 = inception_blk(b4, 128, (128, 256), (24, 64), 64)
    b4 = inception_blk(b4, 112, (144, 288), (32, 64), 64)
    b4 = inception_blk(b4, 256, (160, 320), (32, 128), 128)
    b4 = ZeroPadding2D(padding=(1,1))(b4)
    b4 = MaxPool2D(pool_size=3, strides=2)(b4)

    b5 = inception_blk(b4, 256, (160, 320), (32, 128), 128)
    b5 = inception_blk(b5, 384, (192, 384), (48, 128), 128)
    b5 = GlobalAveragePooling2D()(b5)

    b6 = Dense(10)(b5)

    net = Model(inpt, b6, name='inception')
    return net
