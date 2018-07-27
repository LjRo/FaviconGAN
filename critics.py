import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Conv2DTranspose, UpSampling2D, Conv2D

def build_critic(img_shape):
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

def build_conv_critic(img_shape):
    model = Sequential()

    depth = 32
    dropout = 0.4
    # In: 1 x 28 x 28, depth=1
    # Out: 1 x 10 x 10, depth=64
    #in_shape = (img_shape[2],img_shape[0], img_shape[1])
    in_shape = img_shape
    model.add(Conv2D(depth*1, (5, 5), strides=(2, 2), padding="same", input_shape=in_shape, data_format='channels_last'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(depth*2, (5, 5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(depth*4, (5, 5), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))

    # Out: 1-dim probability
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)