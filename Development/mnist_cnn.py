#Importing Libraries

import mnist
from scipy.misc import imsave
import numpy as np


import tensorflow
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



#Loading dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

from keras.utils.np_utils import to_categorical
y_train_labels = to_categorical(y_train, num_classes=None)
y_test_labels = to_categorical(y_test, num_classes=None)

# reshape to be [samples][pixels][width][height]
X_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

##
# Constructing convolutional neural network
##

# Initializing the CNN
classifier = Sequential()


classifier.add(Convolution2D(filters=32, kernel_size=(4, 4), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


# Flattening
classifier.add(Flatten())


from keras.layers.advanced_activations import LeakyReLU

# Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#Training
classifier.fit(X_train, y_train_labels, epochs=10, batch_size=128)

#Testing
scores = classifier.evaluate(X_test, y_test_labels, verbose=0)
print(scores)


# saving model
import h5py
from keras.models import load_model
classifier.save('my_model.h5')