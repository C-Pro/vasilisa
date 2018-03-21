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
popul_size=30

meta=[1,        #number of conv layers (1-3)
      1, 1, 1,  #first conv layer hyperparameters (filters, kernel, stride)
      0, 0,     #first pooling layer (pool, stride). NO layer if zeros
      0, 0, 0,  #second conv layer hyperparameters (filters, kernel, stride)
      0, 0,     #second pooling layer (pool, stride). NO layer if zeros
      0, 0, 0,  #third conv layer hyperparameters (filters, kernel, stride)
      0, 0,     #third pooling layer (pool, stride). NO layer if zeros
      100       #dense layer neuron count
    ]
#maximum values (-1) for all
#hyperparameters
meta_limits=[3,
             8, 3, 3,
             3, 2,
             4, 3, 3,
             3, 2,
             2, 3, 3,
             2, 2,
             512
        ]

population=[]
losses=[]

def random_meta():
    "Random hyperparameters"
    return [np.random.randint(1, meta_limits[i]+1) for i in range(len(meta))]

def mutate(meta, prob=0.1):
    "Mutate some of hyperparameters"
    return [meta[i] if np.random.rand()>prob else np.random.randint(1, meta_limits[i]+1) \
            for i in range(len(meta))]

def crossover(mother, father, other):
    "Yes, why not have threesome reproduction?"
    return [int(np.mean(np.array([mother[i],father[i],other[i]]))) \
            for i in range(len(meta))]

def save_population(population, losses):
    with open("population.json", "w") as f:
        f.write(json.dumps({"population": population, "losses": losses}, indent=2))

def load_population():
    try:
        with open("population.json", "rt") as f:
            j = json.loads(f.read())
            return j["population"], j["losses"], True
    except Exception as e:
        return None, None, False

min_test_loss=100500
best_meta=meta[:]

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


population, losses, ok = load_population()
if not ok:
    population=[random_meta() for _ in range(popul_size)]
    losses=[100500]*popul_size
popul_loaded=ok

while True:

    #if population loaded from file
    #then skip first evaluation
    if not popul_loaded:
        #evaluating population
        for i in range(popul_size):
            print("Evaluating model ", i)
            curr_meta = population[i]
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
                          batch_size=15,
                          epochs=1,
                          verbose=1) #,
                          #validation_data=(x_test, y_test))
                score = model.evaluate(x_test, y_test, verbose=0)
                losses[i] = score[0]
                print('Test loss:', score[0])
                print('Test accuracy:', score[1])

                #Save best performing model and its meta
                if min_test_loss > score[0]:
                    print("Best model to date!")
                    min_test_loss = score[0]
                    best_meta = curr_meta[:]
                    model.save("vasilisa.model")
                    with open("vasilisa_meta.json", "w") as f:
                        f.write(json.dumps({"meta": best_meta, "loss": score[0], "accuracy": score[1]}))
            except Exception as e:
                print("Caught exception: ", e)
            sess.close()
    else:
        popul_loaded = False

    #replacing worst performing half of population
    #with mutated offsprings of best part
    median_loss = np.median(losses)
    best_half=[i for (i, l) in enumerate(losses) if l <= median_loss]
    worst_half=[i for (i, l) in enumerate(losses) if l > median_loss]
    for i in worst_half:
        parents_i = random.sample(best_half, 3)
        parents = [population[x] for x in parents_i]
        print("Replacing model {} with loss {:.4f} with crossover of ({:.4f},{:.4f},{:.4f})".format(
                i, losses[i], *[losses[x] for x in parents_i]))
        population[i] = mutate(crossover(*parents))
        losses[i] = 100500

    save_population(population, losses)



