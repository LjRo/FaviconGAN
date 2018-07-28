import generator as gen
import critics as crit
import numpy as np
import generator as gen
import critics as crit
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.models import model_from_json

class GAN:
    def __init__(self):
        #Params
        self.input_size = 100
        self.image_shape = (28,28,1)
        self.num_classes = 10
        #Init Generator
        self.generator = gen.build_conv_generator(self.input_size,self.image_shape,self.num_classes)

        self.n_critic = 5
        self.clip_value = 0.01
        self.critic = crit.build_conv_critic(self.image_shape,self.num_classes)
        optimizer = Adam(0.0001,0.5,0.9)
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        
        #Input vector 128
        noise = Input(shape=(self.input_size,))
        label = Input(shape=(1,))

        #Generator init
        img = self.generator([noise, label])
        self.critic.trainable = False
        valid = self.critic([img, label])
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=self.wasserstein_loss,optimizer=optimizer,metrics=['accuracy'])
        

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true*y_pred)

    def __call__(self,epochs, batch_size=64, sample_interval=50):
        #Dataset input
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        #x_train = x_train[0:2000]
        X_train = (x_train.astype(np.float32)-127.5)/127.5
        X_train = np.expand_dims(X_train,axis=3)
        y_train = y_train.reshape(-1, 1)
        print("X_train.shape: ",X_train.shape)
        print("y_train.shape: ", y_train.shape)

        valid = -np.ones((batch_size,1))
        fake = np.ones((batch_size,1))

        for epoch in range(epochs):
            #Critic
            if epoch>400:
                self.n_critic=1
            for _ in range(self.n_critic):
                idx = np.random.randint(0,X_train.shape[0],batch_size)
                imgs, labels = X_train[idx], y_train[idx]

                noise = np.random.normal(0,1,(batch_size,self.input_size))
                gen_imgs = self.generator.predict([noise, labels])

                d_loss_real = self.critic.train_on_batch([imgs, labels],valid)
                d_loss_fake = self.critic.train_on_batch([gen_imgs, labels],fake)

                d_loss = 0.5*np.add(d_loss_fake, d_loss_real)

                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w,-self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
                
            #condition on labels (?)
            sampled_labels = np.random.randint(0,10,batch_size).reshape(-1,1)

            #Generator
            g_loss = self.combined.train_on_batch([noise,sampled_labels],valid)
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                #self.combined.save('my_model.h5')
                #self.generator.save('my_model.h5')
    def sample_images(self,epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.input_size))
        sampled_labels = np.arange(0,10).reshape(-1,1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].set_title("Label: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
plt.close()