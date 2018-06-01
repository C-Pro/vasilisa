#!/usr/bin/env python

from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import InputLayer, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.python.keras.models import Sequential

import glob
import sys
import json
import os
import random

width=200
height=99

#minimal values for hyperparameters
meta_min=[1,        #number of conv layers (1-3)
          1, 1, 1,  #first conv layer hyperparameters (filters, kernel, stride)
          0, 1,     #first pooling layer (pool, stride). NO layer if zeros
          1, 1, 1,  #second conv layer hyperparameters (filters, kernel, stride)
          0, 1,     #second pooling layer (pool, stride). NO layer if zeros
          1, 1, 1,  #third conv layer hyperparameters (filters, kernel, stride)
          0, 1,     #third pooling layer (pool, stride). NO layer if zeros
          0         #dense layer neuron count
         ]

#maximum values for all
#hyperparameters
meta_max=[3,
          32, 4, 4,
          4, 3,
          32, 4, 4,
          4, 3,
          32, 4, 4,
          4, 3,
          4096
          ]

def random_meta():
    "Random hyperparameters"
    return [np.random.randint(meta_min[i], meta_max[i]+1) for i in range(len(meta_min))]

min_test_loss=100500
best_meta=[]

def make_model(meta):
    'create model based on meta definition'
    model = Sequential()
    model.add(InputLayer(input_shape=(width, height, 1)))
    for l in range(meta[0]):
        print("Conv2D({},{},{})".format(meta[1+l*5],
                                       meta[2+l*5],
                                       meta[3+l*5]))
        model.add(Conv2D(meta[1+l*5],
                         kernel_size=meta[2+l*5],
                         strides=meta[3+l*5],
                         activation='relu'))
        if meta[4+l*5] > 0:
            print("MaxPooling2D({},{})".format(meta[4+l*5],
                                       meta[5+l*5]))
            model.add(MaxPooling2D(pool_size=meta[4+l*5],
                                   strides=meta[5+l*5]))

    model.add(Flatten())
    if meta[-1]>0:
        print("Dense({})".format(meta[-1]))
        model.add(Dense(meta[-1], activation='relu'))
    model.add(Dropout(0.75))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    #randomize weights
    weights = model.get_weights()
    weights = [np.random.normal(size=w.shape) for w in weights]
    model.set_weights(weights)

    return model


#load training data
data = []
for y in [0,1]:
    for f in glob.glob('{}/*.png'.format(y)):
        if not os.path.basename(f).startswith("val"):
            data.append(list(Image.open(f).getdata(0)) + [y])

d = np.array(data, np.float32)
np.random.shuffle(d)
x = [f[:width*height] for f in d]
y = [[f[width*height], 1-f[width*height]] for f in d]
x_train = np.array(x)
y_train = np.array(y)
x_train = np.reshape(x_train, (x_train.shape[0], width, height, 1))

print(x_train.shape)
print(y_train.shape)

#load test data
data = []
for y in [0,1]:
    for f in glob.glob('{}/*.png'.format(y)):
        if os.path.basename(f).startswith("val"):
            data.append(list(Image.open(f).getdata(0)) + [y])
d = np.array(data, np.float32)
np.random.shuffle(d)
x = [f[:width*height] for f in d]
y = [[f[width*height], 1-f[width*height]] for f in d]
x_test = np.array(x)
y_test = np.array(y)
x_test = np.reshape(x_test, (x_test.shape[0], width, height, 1))

print(x_test.shape)
print(y_test.shape)


while True:

    curr_meta = random_meta()
    model=None

    try:
        sess = tf.Session()
        tf.set_random_seed(42)
        K.set_session(sess)
        model = make_model(curr_meta)
    except Exception as e:
        print("Bad meta: ", e)
        sess.close()
        continue

    try:
        model.fit(x_train, y_train,
                  batch_size=30,
                  epochs=8,
                  shuffle=True,
                  verbose=1) #,
                  #validation_data=(x_test, y_test))
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

        #Save best performing model and its meta
        if min_test_loss > score[0]:
            print("#### Best model to date! ####")
            min_test_loss = score[0]
            best_meta = curr_meta[:]
            model.save("vasilisa_rnd.model")
            with open("vasilisa_meta_rnd.json", "w") as f:
                f.write(json.dumps({"meta": best_meta, "loss": score[0], "accuracy": score[1]}))

    except Exception as e:
        print("Caught exception: ", e)
    sess.close()


