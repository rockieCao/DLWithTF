# -*- coding: UTF-8 -*-

import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization, \
    GlobalAveragePooling2D, ZeroPadding2D, Input

def residual_blk(inpt, num_channels, strides=1, use_1x1conv=False):
    pad1 = ZeroPadding2D(padding=(1,1))
    conv1 = Conv2D(filters=num_channels, kernel_size=3, strides=strides)
    pad2 = ZeroPadding2D(padding=(1,1))
    conv2 = Conv2D(filters=num_channels, kernel_size=3)
    if use_1x1conv:
        conv3 = Conv2D(filters=num_channels, kernel_size=1, strides=strides)
    else:
        conv3 = None
    bn1 = BatchNormalization()
    bn2 = BatchNormalization()

    blk = Sequential()
    blk.add(pad1)
    blk.add(conv1)
    blk.add(bn1)
    blk.add(pad2)
    blk.add(conv2)
    blk.add(bn2)

    blk2 = inpt
    if conv3:
        blk2 = conv3(blk2)

    x = keras.layers.add([blk(inpt), blk2])
    x = Activation('relu')(x)
    return x


def resnet_blk(inpt, num_channel, num_residual, first_block=False):
    x = inpt
    for i in range(num_residual):
        if i == 0 and not first_block:
            x = residual_blk(inpt=x, num_channels=num_channel, use_1x1conv=True, strides=2)
        else:
            x = residual_blk(inpt=x, num_channels=num_channel)
    return x


# resnet-18
def build_net(input_shape):
    inpt = Input(input_shape)
    x = ZeroPadding2D(padding=(3,3))(inpt)
    x = Conv2D(64, kernel_size=7, strides=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = MaxPool2D(pool_size=3, strides=2)(x)
    x = resnet_blk(x, 64, 2, first_block=True)
    x = resnet_blk(x, 128, 2)
    x = resnet_blk(x, 256, 2)
    x = resnet_blk(x, 512, 2)
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    net = Model(inpt, x, name='resnet')
    return net



