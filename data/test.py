#!/usr/bin/env python3
import tensorflow as tf
import numpy as np

def make_model():

	if 1:
		# With this model use of ModelCheckpoint() in train() saves such that model.load_weights() fails in predict()
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(1))
	else:
		# With this functionally equivalent model it loads without complaint
		inp = tf.keras.layers.Input(shape=(1, ))
		l = tf.keras.layers.Dense(1)(inp)
		model = tf.keras.Model(inputs=inp, outputs=l)

	model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(lr=0.01))
	return model

n = 1
x = np.zeros((n, 1), dtype=np.float32) 
y = np.zeros((n, 1), dtype=np.float32) 

def train():
	model = make_model()
	saver = tf.keras.callbacks.ModelCheckpoint(filepath='test.hdf5', verbose=True, period=1)
	model.fit(x, y, batch_size=1, verbose=1, epochs=1, callbacks=[saver])

def predict():
	model = make_model()
	model.load_weights('test.hdf5');
	# ... would call model.predict()

# build the model, train and save
train()

# build, the model, load the trained weights, and predict
predict()

