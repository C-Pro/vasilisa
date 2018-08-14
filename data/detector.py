#!/usr/bin/env python

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import load_model


import glob
import sys
import json
import os
import io
import time
import subprocess
import threading
import tempfile
import queue

width=200
height=99

THRESHOLD=0.95

rec = "arecord -D default -r 16000 -f S16_LE -d 2 -t raw"
spec = "sox -e signed-integer -t raw -b 16 -r 16000 - -n spectrogram -mr -x 200 -y 99 -o"


def on_activation():
    '''callback function called when activation
    word is detected'''
    print("Василиса")

def record(q):
    'record 2s sample and put it into queue'
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        #print(f.name)
        r = subprocess.Popen(rec.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #print(r.stderr.read())
        #print("Running " + spec + " " + f.name)
        s = subprocess.Popen((spec + " " + f.name).split(" "),
                              stdin=r.stdout,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        s.wait()
        #print("Done")
        #print(s.stderr.read())
        #time.sleep(1)
        data = list(Image.open(f).getdata(0))
    q.put(data)

def detect(q):
    'run detector loop'
    sess = tf.Session()
    K.set_session(sess)
    model = load_model("vasilisa.h5")

    while True:
        data = q.get()
        d = np.array(data, np.float32)
        x = np.reshape(d, (1, width, height, 1))
        p = model.predict(x)
        if p[0][0] >= THRESHOLD:
            on_activation()

if __name__ == "__main__":

    q = queue.Queue()

    detector_thread = threading.Thread(target=detect, args=[q], daemon=True)
    detector_thread.start()

    while True:
            t = threading.Thread(target=record, args=[q])
            t.start()
            time.sleep(1)

