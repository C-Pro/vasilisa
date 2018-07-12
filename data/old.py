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

rec = "arecord -D default -r 16000 -f S16_LE -d 2 -t raw tmpfile.raw".split(" ")
spec = "sox -e signed-integer -t raw -b 16 -r 16000 tmpfile.raw -n spectrogram -mr -x 200 -y 99 -o spectrogram.png".split(" ")

sess = tf.Session()
tf.set_random_seed(42)
K.set_session(sess)

model = load_model("vasilisa.h5", compile=False)

while True:
    subprocess.run(rec, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(spec, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    data = list(Image.open("spectrogram.png").getdata(0))
    d = np.array(data, np.float32)
    x = np.reshape(d, (1, width, height, 1))

    p = model.predict(x)
    if p[0][0] >= 0.97:
        print("Василиса")

