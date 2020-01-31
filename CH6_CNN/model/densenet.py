# -*- coding: UTF-8 -*-

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, AveragePooling2D, ZeroPadding2D, Input

def conv_blk(num_channels):
    blk = Sequential()
    blk.add(BatchNormalization())
    blk.add(Activation('relu'))
    blk.add(ZeroPadding2D(padding=(1, 1)))
    blk.add(Conv2D(num_channels, kernel_size=3))
    return blk


def dense_blk(inpt, num_convs, num_channels):
    x = inpt
    for _ in range(num_convs):
        y = conv_blk(num_channels)(x)
        x = keras.layers.concatenate([x, y], axis=3)
    return x


def transition_block(num_channels):
    blk = Sequential()
    blk.add(BatchNormalization())
    blk.add(Activation('relu'))
    blk.add(Conv2D(num_channels, kernel_size=1))
    #blk.add(AveragePooling2D(pool_size=2, strides=2)) #original setting
    blk.add(AveragePooling2D(pool_size=1, strides=2))
    return blk


def build_net(input_shape):
    inpt = Input(input_shape)

    num_channel_init = 64

    x = ZeroPadding2D(padding=(3,3))(inpt)
    x = Conv2D(num_channel_init, kernel_size=7, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)

    num_channels = [32, 32, 32, 32]
    num_convs = [4, 4, 4, 4]

    num_channel_cur = num_channel_init
    for i, (num_conv, num_channel) in enumerate(zip(num_convs, num_channels)):
        x = dense_blk(x, num_conv, num_channel)
        num_channel_cur += num_conv * num_channel
        if i != len(num_convs) - 1:
            num_channel_cur //= 2
            x = transition_block(num_channel_cur)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    net = Model(inpt, x, name='densenet')
    return net
