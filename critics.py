import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Embedding, multiply
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

def build_conv_critic(img_shape,num_classes):
    model = Sequential()

    depth = 32
    dropout = 0.4

    model.add(Dense(1024, input_dim=np.prod(img_shape)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(BatchNormalization(momentum=0.9))

    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1))

    img = Input(shape=img_shape)
    print ("img.shape ", img.shape)
    label = Input(shape=(1,), dtype='int32')
    print ("label.shape ", label.shape)
    label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))
    print ("label_embedding.shape ", label_embedding.shape)
    flat_img = Flatten()(img)
    print ("flat_img.shape ", flat_img.shape)
    model_input = multiply([flat_img, label_embedding])
    print ("model_input.shape ", model_input.shape)

    validity = model(model_input)

    return Model([img, label], validity)

# def build_conv_critic(img_shape):
#     model = Sequential()

#     depth = 32
#     dropout = 0.4
 
#     in_shape = img_shape
#     model.add(Conv2D(depth*1, (5, 5), strides=(2, 2), padding="same", input_shape=in_shape, data_format='channels_last'))
#     model.add(LeakyReLU(alpha=0.2))

#     model.add(Conv2D(depth*2, (5, 5), strides=(2,2), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))

#     model.add(Conv2D(depth*4, (5, 5), strides=(2,2), padding='same'))
#     model.add(LeakyReLU(alpha=0.2))

#     # Out: 1-dim probability
#     model.add(Flatten())
#     model.add(Dense(1))

#     img = Input(shape=img_shape)
#     validity = model(img)

#     return Model(img, validity)