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
import dataset as data
from time import time
from time import gmtime, strftime
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.callbacks import TensorBoard
from keras.models import model_from_json
from keras.models import load_model
from PIL import Image

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names,logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value= value
        summary_value.tag = name
        callback.writer.add_summary(summary,batch_no)
        callback.writer.flush()

class GAN:
    def __init__(self):

        tf.reset_default_graph()
        log_dir = "./logs/"

        writer = tf.summary.FileWriter(os.path.join(log_dir,'faviconGAN'))
        #Params
        self.input_size = 128
        self.image_shape = (32,32,3)
        self.num_classes = 50
        #Init Generator
        self.generator = gen.build_conv_generator(self.input_size,self.image_shape,self.num_classes)
        self.save_epoch = 5000
        self.n_critic = 5
        self.clip_value = 0.01
        self.critic = crit.build_conv_critic(self.image_shape,self.num_classes)
        optimizer = Adam(0.0001,0.5,0.9)
        self.critic.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

        #Input vector 128
        noise = Input(shape=(self.input_size,))
        label = Input(shape=(1,))

        #Generator init
        img = self.generator([noise, label])
        valid = self.critic([img, label])
        self.critic.trainable = False
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=self.wasserstein_loss,optimizer=optimizer,metrics=['accuracy'])
        writer.add_graph(tf.get_default_graph())

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true*y_pred)

    def __call__(self,epochs, batch_size=64, sample_interval=50):
        #Dataset input
        #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        (X_train, Y_train) = data.getIcons50ClassDataset()
        #X_train = np.expand_dims(X_train,axis=3)
        Y_train = Y_train.reshape(-1, 1)
        print("X_train.shape: ",X_train.shape)
        print("Y_train.shape: ", Y_train.shape)

        valid = -np.ones((batch_size,1))
        fake = np.ones((batch_size,1))
    
        log_dir = "./logs/"
        callback = TensorBoard(log_dir)
        callback2 = TensorBoard(log_dir)
        callback.set_model(self.critic)
        callback2.set_model(self.generator)
        train_names = ['d_loss']
        train_names2 = ['g_loss']
        for epoch in range(epochs):
            #Critic

            if epoch>1000:
                self.n_critic=1
            for _ in range(self.n_critic):
                idx = np.random.randint(0,X_train.shape[0],batch_size)
                imgs, labels = X_train[idx], Y_train[idx]

                noise = np.random.normal(0,1,(batch_size,self.input_size))
                gen_imgs = self.generator.predict([noise, labels])

                d_loss_real = self.critic.train_on_batch([imgs, labels],valid)
                d_loss_fake = self.critic.train_on_batch([gen_imgs, labels],fake)

                self.d_loss = 0.5*np.add(d_loss_fake, d_loss_real)
                if epoch % 50 == 0:
                    write_log(callback,train_names,self.d_loss,epoch)

                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w,-self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
            #condition on labels (?)
            sampled_labels = np.random.randint(0,50,batch_size).reshape(-1,1)

            #Generator
            self.g_loss = self.combined.train_on_batch([noise,sampled_labels],valid)
            if epoch % 50 == 0:
                write_log(callback2,train_names2,self.g_loss,epoch)

            if epoch % self.save_epoch == 0:
                self.combined.save("models\combined_"+str(epoch)+".model")
                self.generator.save("models\generator"+str(epoch)+".model")
                self.critic.save("models\critic_"+str(epoch)+".model")

            print("%d [D loss: %f] [G loss: %f]" % (epoch, self.d_loss[0], self.g_loss[0]))
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def generate(self,gen_model,x,y):
        optimizer = Adam(0.0001,0.5,0.9)
        gen_model = load_model(gen_model)
        gen_model.compile(loss=self.wasserstein_loss, optimizer=optimizer, metrics=['accuracy'])
        self.generator = gen_model
        #Input vector 128

        
        self.sample_images(strftime("%H-%M-%S", gmtime()),x,y)

    def sample_images(self,epoch,r=5,c=10):
        noise = np.random.normal(0, 1, (r * c, self.input_size))
        sampled_labels = np.arange(0,50).reshape(-1,1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        #print(gen_imgs[0][:][:][:])
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].set_title("L: %d" % sampled_labels[cnt])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/icon_%s.png" % epoch)
        plt.close()

