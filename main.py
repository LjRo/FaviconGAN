import wgan
import sys
import os

if(len(sys.argv) < 2):
    wgan = wgan.GAN()
    wgan(100000,128,1000)
elif (len(sys.argv) == 5):
    if(sys.argv[1] == "load"):
        model_name = sys.argv[2]
        x = int(sys.argv[3])
        y = int(sys.argv[4])
        model = os.path.join("models",model_name)
        wgan = wgan.GAN()
        wgan.generate(model,x,y)
elif (len(sys.argv) == 4):
    if(sys.argv[1] == "load"):
        critic_model = sys.argv[2]
        gen_model = sys.arvg[1]