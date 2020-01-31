# -*- coding: utf-8 -*-

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization

net = Sequential()
net.add(Conv2D(5, kernel_size=11, strides=1, activation='relu', name='input_image'))
net.add(MaxPool2D(pool_size=2, strides=2))
net.add(Conv2D(10, kernel_size=5, padding='Same', activation='relu'))
net.add(MaxPool2D(pool_size=2, strides=2))
# 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
# 前两个卷积层后不使用池化层来减小输入的高和宽
net.add(Conv2D(30, kernel_size=3, padding='Same', activation='relu'))
net.add(Conv2D(30, kernel_size=3, padding='Same', activation='relu'))
net.add(Conv2D(10, kernel_size=3, padding='Same', activation='relu'))
net.add(MaxPool2D(pool_size=2, strides=2))
# 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
net.add(Flatten())
net.add(Dense(90, activation="relu"))
net.add(Dropout(0.5))
net.add(Dense(10, activation="softmax", name='output_class'))
