import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution2D, Conv2DTranspose, UpSampling2D, Conv2D

#nije prilagodjen conditional ganu
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

#jeste prilagodjen conditional ganu
def build_conv_generator(input_size,img_shape,num_classes): 
    channels = img_shape[2]
    dropout = 0.4
    depth = 128
    dim = 7

    model = Sequential()

    model.add(Dense(dim*dim*depth, input_dim=input_size))
    model.add(Activation('relu'))
    model.add(Reshape((dim, dim, depth)))
  
    model.add(Conv2DTranspose(int(depth/2), (5, 5),strides=(2,2), padding='same',output_shape=(None, 2*dim, 2*dim, int(depth/2)), data_format='channels_last'))
    model.add(BatchNormalization(momentum=0.9))
    model.add(Activation('relu'))    

    # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
    model.add(Conv2DTranspose(1, (5, 5),strides=(2,2), padding='same',output_shape=(None, 4*dim, 4*dim, 1), data_format='channels_last'))
    model.add(Activation('tanh'))

    noise = Input(shape=(input_size,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(num_classes, input_size)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)

    return Model([noise, label], img)
    
