import pickle
import math
import os
import h5py
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from skimage import io
import util

model = None
json_filename = "model.json"
h5_filename = "model.h5"
with open(json_filename, 'r') as jfile:
    model = model_from_json(json.load(jfile))
model.compile("adam", "mse")
model.load_weights(h5_filename)

X_test = []
image = plt.imread("./sample-data/data/IMG/center_2016_12_01_13_30_48_287.jpg")
X_test.append(image)
image = plt.imread("./sample-data/data/IMG/center_2016_12_01_13_31_13_279.jpg")
X_test.append(image)
image = plt.imread("./sample-data/data/IMG/center_2016_12_01_13_32_44_569.jpg")
X_test.append(image)
image = plt.imread("./sample-data/data/IMG/center_2016_12_01_13_33_12_094.jpg")
X_test.append(image)
image = plt.imread("./sample-data/data/IMG/center_2016_12_01_13_33_09_562.jpg")
X_test.append(image)
image = plt.imread("./sample-data/data/IMG/center_2016_12_01_13_33_27_063.jpg")
X_test.append(image)
image = plt.imread("./track1-opencorner2/IMG/center_2017_01_25_16_28_02_613.jpg")
X_test.append(image)
image = plt.imread("./track1-opencorner2/IMG/center_2017_01_25_16_28_01_775.jpg")
X_test.append(image)
image = plt.imread("./track1-normal1/IMG/center_2017_01_24_21_32_27_913.jpg")
X_test.append(image)

f, axarr = plt.subplots(len(X_test), 2)
for i in range(len(X_test)):
    image = util.preprocess_image(X_test[i])
    #image = util.flip_image(image)
    input = np.array([image])
    steering_angle = 0.0
    steering_angle = float(model.predict(input, batch_size=1))
    print ("axarr", axarr)
    print ("shape", image.shape)
    axarr[i][0].set_title(steering_angle)
    axarr[i][0].imshow(X_test[i])
    axarr[i][1].imshow(image[:,:,0], cmap='gray')
    plt.setp(axarr[i][0].get_xticklabels(), visible=False)
    plt.setp(axarr[i][0].get_yticklabels(), visible=False)
    plt.setp(axarr[i][1].get_xticklabels(), visible=False)
    plt.setp(axarr[i][1].get_yticklabels(), visible=False)
plt.show()
