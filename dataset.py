import numpy as np
import os
from PIL import Image


def getIcons50ClassDataset():
    train_path = '.\\icons50\\'
    class_batch = os.listdir(train_path)
    x_train = []
    y_train = []
    Y_train = []
    i = 0
    for class_name in class_batch:
        class_path = os.path.join(train_path,class_name)
        train_batch = os.listdir(class_path)
        i += 1
        print(i,'/',len(class_batch))
        for sample in train_batch:
            img_path = os.path.join(class_path,sample)
            try:
                im = np.asarray(Image.open(img_path))
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
            x_train.append(im)
            if not(class_name in y_train):
                y_train.append(class_name)
                Y_train.append(y_train.index(class_name))
            else:
                Y_train.append(y_train.index(class_name))
            
    
    x_train=np.array(x_train)
    X_train = (x_train.astype(np.float32)-127.5)/127.5
    Y_train = np.array(Y_train)
    print(X_train.shape,' - ', Y_train.shape)
    return (X_train, Y_train)

def getFaviconsDataset():
    train_path = '.\\filtered\\'
    train_batch = os.listdir(train_path)
    x_train = []
    #train_batch = train_batch[0:2000]
    # if data are in form of images
    i = 0
    for sample in train_batch:
        img_path = train_path+sample
        print(i,'/',len(train_batch))
        i += 1
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
    
    x_train=np.array(x_train)
    X_train = (x_train.astype(np.float32)-127.5)/127.5
    return (X_train,[])