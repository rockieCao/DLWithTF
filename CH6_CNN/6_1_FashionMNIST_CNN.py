# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

IMG_ROWS = 28
IMG_COLS = 28
NUM_CLASSES = 10
VALID_SIZE = 0.2
RANDOM_STATE = 456

np.random.seed(RANDOM_STATE)

PATH = "../data/FashionMNIST"
train_file = os.path.join(PATH, "fashion-mnist_train.csv")
test_file = os.path.join(PATH, "fashion-mnist_test.csv")

train_data = pd.read_csv(train_file)
print("train data:", train_data.shape)
test_data = pd.read_csv(test_file)
print("test data:", test_data.shape)

# Create a dictionary for each type of label
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}

def get_label_distribution(data, needPlot=False):
    label_counts = data['label'].value_counts()
    total_samples = len(data)
    for i in range(label_counts):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{:<20s}: {} or {}%".format(label, count, percent))
    if needPlot:
        f, ax = plt.subplot(1,1, figsize=(12,4))
        g = sns.countplot(data.label, order=data['label'].value_counts().index)
        g.set_title("Number of labels for each class")
        for p, label in zip(g.patches, data["label"].value_counts().index):
            g.annotate(labels[label], (p.get_x(), p.get_height()+0.1))
        plt.show()


def sample_images_data(data):
    sample_images = []
    sample_labels = []
    for k in labels.keys():
        samples = data[data['label']==k].head(4)
        for j,s in enumerate(samples.values):
            img = np.array(samples.iloc[j, 1:]).reshape(IMG_ROWS, IMG_COLS)
            sample_images.append(img)
            sample_labels.append(samples.iloc[j, 0])
    print("Total number of sample images to plot: ", len(sample_images))
    return sample_images, sample_labels


def plot_sample_images(data_sample_images, data_sample_labels, cmap="Blues"):
    f, ax = plt.subplots(5,8, figsize=(16,10))
    for i,img in enumerate(data_sample_images):
        ax[i//8, i%8].imshow(img, cmap=cmap)
        ax[i//8, i%8].axis('off')
        ax[i//8, i%8].set_title(labels[data_sample_labels[i]])
    plt.show()


def data_preprocessing(data):
    y = pd.get_dummies(data.values[:, 0]).values
    num_images = data.shape[0]
    x_as_array = data.values[:, 1:]
    x_reshape = x_as_array.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    X = x_reshape / 255
    return X, y


def plot_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    g = sns.countplot(ydf[0], order=np.arange(0, 10))
    g.set_title("Number of items for each class")
    g.set_xlabel("Category")

    for p, label in zip(g.patches, np.arange(0, 10)):
        g.annotate(labels[label], (p.get_x(), p.get_height() + 0.1))

    plt.show()


def get_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    # Get the count for each label
    label_counts = ydf[0].value_counts()

    # Get total number of samples
    total_samples = len(yd)

    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = labels[label_counts.index[i]]
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{:<20s}:   {} or {}%".format(label, count, percent))


def plot_train_vlad_curve(history):
    # plot training / validation loss & acc
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.suptitle('Train results', fontsize=10)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['acc'], color='green', label='Training Accuracy')
    plt.plot(history.history['val_acc'], color='orange', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()


#train_sample_images, train_sample_labels = sample_images_data(train_data)
#plot_sample_images(train_sample_images, train_sample_labels, "Greens")

# print("Label distribution in training data:")
# get_label_distribution(train_data, needPlot=True)

X_raw, y_raw = data_preprocessing(train_data)
X_test, y_test = data_preprocessing(test_data)

X_train, X_val, y_train, y_val = train_test_split(X_raw, y_raw, test_size=VALID_SIZE, random_state=RANDOM_STATE)
print("Fashion MNIST train - rows:", X_train.shape[0], " columns:", X_train.shape[1:4])
print("Fashion MNIST valid - rows:", X_val.shape[0], " columns:", X_val.shape[1:4])
print("Fashion MNIST test - rows:", X_test.shape[0], " columns:", X_test.shape[1:4])

print("y_train.shape:", y_train.shape)
print(np.argmax(y_train, axis=1))

#plot_count_per_class(np.argmax(y_train, axis=1))
#get_count_per_class(np.argmax(y_train, axis=1))

needTSNE = False
if needTSNE:
    from sklearn.utils import shuffle

    tsne_obj = TSNE(n_components=2,
                    init='pca',
                    random_state=RANDOM_STATE,
                    method='barnes_hut',
                    n_iter=250,
                    verbose=2)
    index_input = list(range(len(X_val)))
    index_sample = shuffle(index_input, random_state=RANDOM_STATE)[:1000]
    print(index_sample)
    X_val_sample = X_val[index_sample]
    y_val_sample = y_val[index_sample]
    tsne_features = tsne_obj.fit_transform(np.reshape(X_val_sample, [X_val_sample.shape[0], -1]))

    obj_categories = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']
    colors = plt.cm.rainbow(np.linspace(0, 1, 10))
    plt.figure(figsize=(10, 10))

    for c_group, (c_color, c_label) in enumerate(zip(colors, obj_categories)):
        print("c_group:", c_group)
        print("c_color:", c_color)
        print("c_label:", c_label)
        plt.scatter(tsne_features[np.where(y_val_sample == c_group), 0],
                    tsne_features[np.where(y_val_sample == c_group), 1],
                    marker='o',
                    color=c_color,
                    linewidth='1',
                    alpha=0.8,
                    label=c_label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('t-SNE on Testing Samples')
    plt.legend(loc='best')
    plt.savefig('clothes-dist.png')
    plt.show(block=False)

import keras
#keras.backend.set_image_dim_ordering('th')
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Conv2D, Activation, MaxPool2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False, # set input mean to 0 over the dataset
    samplewise_center=False, # set each sample mean to 0
    featurewise_std_normalization=False, # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False, # dimesion reduction
    rotation_range=0.1, # randomly rotate image in the range
    zoom_range=0.1, # randomly zoom image
    width_shift_range=0.1, # randomly shift image horizontally
    height_shift_range=0.1, # randomly shift image vertically
    horizontal_flip=False, # randomly flip images
    vertical_flip=False # randomly flip images
)

datagen.fit(X_train)

#Model
input_shape = [None,28,28,1]

NO_EPOCHS = 50
BATCH_SIZE = 128

#optimizer = RMSprop(lr=.001, rho=0.9, epsilon=1e-08, decay=0.0)
optimizer = Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#region LeNet
from dlexp.model import lenet

needTrainLenet = False
needTestLenet = False

if needTrainLenet:
    lenet.net.build(input_shape)
    lenet.net.summary()
    lenet.net.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    #checkpoint = ModelCheckpoint(filepath='lenet_model.h5', verbose=1, save_best_only=True)

    log_filepath = 'checkpoint/log_lenet'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_lenet.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]
    #datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
    history = lenet.net.fit_generator(X_train, y_train,
                                          shuffle=True,
                                          epochs=NO_EPOCHS,
                                          validation_data=(X_val, y_val),
                                          verbose=2,
                                          steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                          callbacks=cbks
                                          #callbacks=[checkpoint]
                                          )

    # same model
    #model_json = lenet.net.to_json()
    #lenet.net.save_weights('lenet_model.h5', save_format='h5')
    #lenet.net.load_weights('lenet_model.h5')
    lenet.net.save('checkpoint/lenet_model.h5')
    #model = tf.keras.models.load_model('lenet_model.h5')

    # plot training / validation loss & acc
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.subtitle('Train results', fontsize=10)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['acc'], color='green', label='Training Accuracy')
    plt.plot(history.history['val_acc'], color='orange', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

if needTestLenet:
    new_model = keras.models.load_model('checkpoint/lenet_model.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('lenet', 100*acc))

#endregion

#region AlexNet
from dlexp.model import alexnet

needTrainAlexNet = False
needTestAlexNet = False
if needTrainAlexNet:
    alexnet.net.build(input_shape)
    alexnet.net.summary()
    alexnet.net.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # checkpoint = ModelCheckpoint(filepath='alexnet_model.h5', verbose=1, save_best_only=True)

    log_filepath = 'checkpoint/log_alexnet'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_alexnet.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]

    history = alexnet.net.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                      shuffle=True,
                                      epochs=NO_EPOCHS,
                                      validation_data=(X_val, y_val),
                                      verbose=2,
                                      steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                      callbacks=cbks
                                      # callbacks=[checkpoint]
                                      )

    # save model
    alexnet.net.save('checkpoint/alexnet_model.h5')
    # model = tf.keras.models.load_model('alexnet_model.h5')

    # plot training / validation loss & acc
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.suptitle('Train results', fontsize=10)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss', fontsize=16)
    plt.plot(history.history['loss'], color='b', label='Training Loss')
    plt.plot(history.history['val_loss'], color='r', label='Validation Loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.ylabel('Accuracy', fontsize=16)
    plt.plot(history.history['acc'], color='green', label='Training Accuracy')
    plt.plot(history.history['val_acc'], color='orange', label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

if needTestAlexNet:
    new_model = keras.models.load_model('checkpoint/alexnet_model.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('alexnet', 100*acc))

#endregion

#region NiNet
from dlexp.model import nin

needTrainNin = False
needTestNin = False

optimizer = Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

if needTrainNin:
    nin.net.build(input_shape)
    nin.net.summary()

    nin.net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    log_filepath = 'checkpoint/log_nin'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_nin.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]

    history = nin.net.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                        shuffle=True,
                                        epochs=NO_EPOCHS,
                                        validation_data=(X_val, y_val),
                                        verbose=2,
                                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                        callbacks=cbks
                                        # callbacks=[checkpoint]
                                        )

    # save model
    nin.net.save('checkpoint/nin_model.h5')
    # model = tf.keras.models.load_model('nin_model.h5')

    plot_train_vlad_curve(history)


if needTestNin:
    new_model = keras.models.load_model('checkpoint/cp_nin.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('nin', 100 * acc))

#endregion

#region VGG
from dlexp.model import vgg

needTrainVGG = False
needTestVGG = False

optimizer = Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

if needTrainVGG:
    vgg.net.build(input_shape)
    vgg.net.summary()

    vgg.net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    log_filepath = 'checkpoint/log_vgg'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_vgg.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]

    history = vgg.net.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                        shuffle=True,
                                        epochs=NO_EPOCHS,
                                        validation_data=(X_val, y_val),
                                        verbose=2,
                                        steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                        callbacks=cbks
                                        # callbacks=[checkpoint]
                                        )

    # save model
    vgg.net.save('checkpoint/vgg_model.h5')
    # model = tf.keras.models.load_model('vgg_model.h5')

    plot_train_vlad_curve(history)


if needTestVGG:
    new_model = keras.models.load_model('checkpoint/cp_vgg.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('vgg', 100 * acc))


#endregion

#region Inception
from dlexp.model import inception

needTrainInc = False
needTestInc = False

if needTrainInc:
    net = inception.build_net([28,28,1])
    net.summary()
    optimizer = Adam(lr=.005, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    log_filepath = 'checkpoint/log_inception'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_inception.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]

    history = net.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                    shuffle=True,
                                    epochs=NO_EPOCHS,
                                    validation_data=(X_val, y_val),
                                    verbose=2,
                                    steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                    callbacks=cbks
                                    # callbacks=[checkpoint]
                                    )

    # save model
    net.save('checkpoint/inception_model.h5')
    # model = tf.keras.models.load_model('nin_model.h5')

    plot_train_vlad_curve(history)

if needTestInc:
    new_model = keras.models.load_model('checkpoint/cp_inception.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('nin', 100 * acc))

#endregion

#region ResNet
from dlexp.model import resnet

needTrainResnet = False
needTestResnet = False

if needTrainResnet:
    net = resnet.build_net([28, 28, 1])
    net.summary()
    optimizer = Adam(lr=.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=["accuracy"])

    log_filepath = 'checkpoint/log_resnet'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_resnet.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]

    history = net.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                shuffle=True,
                                epochs=NO_EPOCHS,
                                validation_data=(X_val, y_val),
                                verbose=2,
                                steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                callbacks=cbks
                                # callbacks=[checkpoint]
                                )

    # save model
    net.save('checkpoint/resnet_model.h5')
    # model = tf.keras.models.load_model('nin_model.h5')

    plot_train_vlad_curve(history)

if needTestResnet:
    new_model = keras.models.load_model('checkpoint/cp_resnet.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('nin', 100 * acc))

#endregion

#region DenseNet
from dlexp.model import densenet

needTrainDensenet = True
needTestDensenet = True

if needTrainDensenet:
    net = densenet.build_net([28,28,1])
    net.summary()
    optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon=None, decay=0, amsgrad=False)
    net.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    log_filepath = 'checkpoint/log_densenet'
    tb_cb = keras.callbacks.TensorBoard(log_dir=log_filepath, write_images=1, histogram_freq=1)
    cp_filepath = 'checkpoint/cp_densenet.h5'
    cp_cb = ModelCheckpoint(filepath=cp_filepath, monitor='val_acc', mode='max', save_best_only='True')
    cbks = [tb_cb, cp_cb]

    history = net.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                                shuffle=True,
                                epochs=NO_EPOCHS,
                                validation_data=(X_val, y_val),
                                verbose=2,
                                steps_per_epoch=X_train.shape[0] // BATCH_SIZE,
                                callbacks=cbks
                                # callbacks=[checkpoint]
                                )

    # save model
    net.save('checkpoint/densenet_model.h5')
    # model = tf.keras.models.load_model('nin_model.h5')

    plot_train_vlad_curve(history)

if needTestDensenet:
    new_model = keras.models.load_model('checkpoint/cp_densenet.h5')
    new_model.summary()

    new_model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    # Evaluate the restored model.
    loss, acc = new_model.evaluate(X_test, y_test)
    print("Restored model {}, accuracy: {:5.2f}%".format('nin', 100 * acc))

#endregion
