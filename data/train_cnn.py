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

width=200
height=99

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

sess = tf.Session()
tf.set_random_seed(0)
K.set_session(sess)

model = Sequential()
model.add(InputLayer(input_shape=(width, height, 1)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2),
                 activation='relu'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.75))
model.add(Dense(2, activation='softmax'))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=15,
          epochs=2,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
model.save("vasilisa.model")
