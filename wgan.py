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

        #img = self.generator.predict(np.random.rand(1,128))
        image = tf.reshape(self.generator.predict(np.random.rand(1,128)),[-1,32,32,3])
        (self.x_train, y_train), (self.x_test, self.y_test) = tf.keras.datasets.fashion_mnist.load_data()
        with tf.Session() as sess:
            print(image)
            image*=255+127.5
            data = image.dtype('unit8')
            plt.imshow(data)
