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
    dropout = 0.4
    depth = 128
    dim = 4 

    model = Sequential()

    model.add(Dense(dim*dim*depth, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Reshape((dim, dim, depth)))
    
    model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))    
    model.add(Conv2D(128, (5, 5), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))    
    model.add(Conv2D(64, (5, 5), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D(size=(2, 2), data_format="channels_last"))    
    model.add(Conv2D(channels, (5, 5), padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("tanh"))

    #model.add(Conv2DTranspose(int(depth/2), (5, 5),strides=(2,2), padding='same',output_shape=(None, 2*dim, 2*dim, int(depth/2)), data_format='channels_last'))
    #model.add(BatchNormalization(momentum=0.9))
    #model.add(Activation('relu'))    

    #model.add(Conv2DTranspose(int(depth/4), (5, 5),strides=(2,2), padding='same',output_shape=(None, 4*dim, 4*dim, int(depth/2)), data_format='channels_last'))
    #model.add(Activation('relu'))  
    #   
    #model.add(Conv2DTranspose(channels, (5, 5),strides=(2,2), padding='same',output_shape=(None, 8*dim, 8*dim, channels), data_format='channels_last'))
    #model.add(Activation('tanh'))

    noise = Input(shape=(input_size,))
    img = model(noise)

    return Model(noise, img)
    
