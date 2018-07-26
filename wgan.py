import generator as gen
import critics as crit
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.datasets import mnist

class GAN:
    def __init__(self):

        self.input_shape = 128
        self.image_shape = (32,32,3)

        self.generator = gen.build_generator(self.input_shape,self.image_shape)
        self.critic = crit.build_critic(self.image_shape)
        self.optimizer = Adam(0.0002,0.5)
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])

        (self.x_train, y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()

