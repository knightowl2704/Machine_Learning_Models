import numpy as np
import pandas as pd
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)
input_shape = (28, 28, 1)
x_test = x_test.astype("float32")
x_train = x_train.astype("float32")

x_train = x_train/255
x_test = x_test/255

from keras.models import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, Dropout


model = Sequential()
model.add(Conv2D(28,(3,3),input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(10,activation="softmax"))


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x = x_train,y = y_train, epochs= 10)

model.evaluate(x_test,y_test)
