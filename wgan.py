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
from time import time
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from PIL import Image

class GAN:
    def __init__(self):

        tf.reset_default_graph()
        log_dir = "./logs/"

        writer = tf.summary.FileWriter(os.path.join(log_dir,'faviconGAN'))
        #Params
        self.input_size = 128
        self.image_shape = (32,32,3)
        #Init Generator
        self.generator = gen.build_conv_generator(self.input_size,self.image_shape)

        self.n_critic = 5
        self.clip_value = 0.01
        self.critic = crit.build_conv_critic(self.image_shape)
        optimizer = Adam(0.0001,0.5,0.9)
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
        #Input vector 128
        z = Input(shape=(self.input_size,))

        #Generator init
        img = self.generator(z)
        valid = self.critic(img)
        self.critic.trainable = False
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,optimizer=optimizer,metrics=['accuracy'])
        writer.add_graph(tf.get_default_graph())

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true*y_pred)

    def __call__(self,epochs, batch_size=64, sample_interval=50):
        #Dataset input
        #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        train_path = '.\\filtered\\'
        train_batch = os.listdir(train_path)
        x_train = []
        train_batch = train_batch[0:2000]
        # if data are in form of images
        for sample in train_batch:
            img_path = train_path+sample
            try:
                im = np.asarray(Image.open(img_path))
                #print(im.shape)
                if(len(im.shape) > 2):
                    if(im.shape[2] > 3):
                        im = np.delete(im,(3),axis=2)
                    if(im.shape[2] == 2):
                        continue
                else:
                    im = np.array([im,im,im]).T
            except OSError:
                continue
            except ValueError:
                continue
            # preprocessing if required
            x_train.append(im)
            #print(im.shape)
        
        #x_train = x_train[0:2000]
        #x_train[:][:][3] = None
        #layer_one = np.array(x_train[:][:][:][0])
        #layer_two = np.array(x_train[:][:][1])
        #layer_three = np.array(x_train[:][:][2])
        #print(layer_one.shape)
        print(len(x_train))
        print(x_train[0].shape)
        x_train=np.array(x_train)
        X_train = (x_train.astype(np.float32)-127.5)/127.5
        #X_train = np.expand_dims(X_train,axis=3)
        print(X_train.shape)

        valid = -np.ones((batch_size,1))
        fake = np.ones((batch_size,1))

        for epoch in range(epochs):
            #Critic

            if epoch>400:
                self.n_critic=1
            for _ in range(self.n_critic):
                idx = np.random.randint(0,X_train.shape[0],batch_size)
                imgs = X_train[idx]

                noise = np.random.normal(0,1,(batch_size,self.input_size))
                gen_imgs = self.generator.predict(noise)

                d_loss_real = self.critic.train_on_batch(imgs,valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs,fake)

                d_loss = 0.5*np.add(d_loss_fake, d_loss_real)

                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w,-self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
                
            #Generator
            g_loss = self.combined.train_on_batch(noise,valid)
            print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss[0]))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
    def sample_images(self,epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.input_size))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 1
        print(gen_imgs.shape)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
plt.close()