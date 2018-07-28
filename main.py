import wgan
import sys
from keras.models import load_model
import os

if(len(sys.argv) < 2):
    wgan = wgan.GAN()
    wgan(40000,128,500)
elif (len(sys.argv) == 3):
    if(sys.argv[1] == "load"):
        model_name = sys.argv[2]
        model = load_model(os.path.join("models",model_name))
        wgan = wgan.GAN()
        wgan.generate(model)
elif (len(sys.argv) == 4):
    if(sys.argv[1] == "load"):
        critic_model = sys.argv[2]
        gen_model = sys.arvg[1]