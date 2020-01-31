# -*- coding: utf-8 -*-

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization

net = Sequential()

net.add(Conv2D(filters=5, kernel_size=5, activation='sigmoid', padding='Same', name='input_image'))
net.add(MaxPool2D(pool_size=2, strides=2)),
net.add(Conv2D(filters=5, kernel_size=5, activation='sigmoid', padding='Same')),
net.add(MaxPool2D(pool_size=2, strides=2)),
net.add(Flatten())
net.add(Dense(10, activation='softmax', name='output_class'))
