import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Conv2DTranspose, UpSampling2D, Conv2D


def build_generator(input_size,img_shape):
    model = Sequential()
    model.add(Dense(512, input_dim=input_size))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    noise = Input(shape=(input_size,))
    img = model(noise)

    return Model(noise, img)

def build_conv_generator(input_size,img_shape): 
    channels = img_shape[2]
    
    model = Sequential()
    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=input_size))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Conv2D(128, (4,4), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, (4,4), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, (4,4), padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(input_size,))
    img = model(noise)

    return Model(noise, img)
    
