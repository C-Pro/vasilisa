#!/usr/bin/env python

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.callbacks import TensorBoard


import glob
import sys
import json
import os
import time
import subprocess

width=200
height=99

spec = "sox -e signed-integer -t raw -b 16 -r 16000 tmpfile.raw -n spectrogram -mr -x 200 -y 99 -o spectrogram.png".split(" ")

sess = tf.Session()
tf.set_random_seed(42)
K.set_session(sess)

model = load_model("vasilisa.h5", compile=False)

threshold=0.9
THRESHOLD_STEP=0.005

tpr=0
fpr=1

while (fpr!=0) and (threshold < 1):
    n = 0.0
    fp = 0.0
    for f in glob.glob('0/*.raw'):
        n+=1
        spec[9] = f
        subprocess.run(spec)
        data = list(Image.open("spectrogram.png").getdata(0))
        d = np.array(data, np.float32)
        x = np.reshape(d, (1, width, height, 1))
        p = model.predict(x)
        if p[0][0] >=threshold:
            fp += 1.0
    fpr = fp/n

    n = 0.0
    tp = 0.0
    for f in glob.glob('1/*.raw'):
        n+=1
        spec[9] = f
        subprocess.run(spec)
        data = list(Image.open("spectrogram.png").getdata(0))
        d = np.array(data, np.float32)
        x = np.reshape(d, (1, width, height, 1))
        p = model.predict(x)
        if p[0][0] >= threshold:
            tp += 1.0
    tpr = tp / n
    print(f'T: {threshold} TPR: {tpr} FPR: {fpr}')
    threshold += THRESHOLD_STEP

