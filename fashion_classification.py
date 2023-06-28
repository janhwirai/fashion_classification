#Neural network in Keras for image classification problem


import tensorflow as tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fm=keras.datasets.fashion_mnist

(X_train,y_train), (X_test,y_test) = fm.load_data()

plt.matshow(X_train[0])

#Normalize training data before training the neural net
X_train = X_train/255 #16*16
X_test = X_test/255  #Data scaling

#Build the sequential model and add layers into it
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

model=Sequential()

model.add(Flatten(input_shape=[28, 28]))#flatten will convert 2d into 1d 
model.add(Dense(20, activation="relu"))#call add function
model.add(Dense(10, activation="softmax"))#Dense layer is the hidden layer

print(model.summary())

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(X_train, y_train)

model.evaluate(X_test, y_test)

plt.matshow(X_test[0])

yp = model.predict(X_test)

np.argmax(yp[0])

class_labels=["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

print(class_labels[np.argmax(yp[0])])
