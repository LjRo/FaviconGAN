import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, Conv2DTranspose


def create_generator():
    model = Sequential()
    return model

model = create_generator()
print(model)