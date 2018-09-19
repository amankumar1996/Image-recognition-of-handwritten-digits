import matplotlib
import tensorflow
import keras
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import numpy as np


##
# Constructing convolutional neural network
##

# Initializing the CNN
classifier = Sequential()

classifier.add(Convolution2D(filters=32, kernel_size=(4, 4), input_shape=(28,28,1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(filters=32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))


#Flattening
classifier.add(Flatten())

from keras.layers.advanced_activations import LeakyReLU

#Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=10, activation='softmax'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# Part 2: Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory(
    'mnist_dataset/training_set',
    color_mode='grayscale',
    target_size=(28, 28),
    batch_size=128)

test_set = test_datagen.flow_from_directory(
    'mnist_dataset/test_set',
    color_mode='grayscale',
    target_size=(28, 28),
    batch_size=64)

classifier.fit_generator(training_set,
                         steps_per_epoch=60000,
                         epochs=4,
                         validation_data=test_set,
                         validation_steps=10000)
#saving the model
classifier.save('mnist_model_2.h5')

#testing
score = classifier.evaluate_generator(test_set, steps= 10000)
import h5py


from keras.models import load_model

model = load_model('mnist_model_2.h5')
s = model.evaluate_generator(test_set, steps = 100)

'''
from keras.utils import plot_model
import pydot_ng as pydot
plot_model(model, to_file='model.png')

import cv2

jpgfile = cv2.imread("E:\Desktop\Semester 6\Deep learning\Deep_learning_project\mnist_dataset\\test_set\\0\zero_0.png")
jpgfile1 = jpgfile[:,:,1]

prediction = model.predict(np.array(jpgfile1))
myfile = cv2.imread('my_data.png')
myfile = np.array(myfile)
myfile.resize((28,28,1))
model2 = load_model('my_model.h5')
prediction = model.predict(myfile.reshape(1,28,28,1))

cv2.imwrite('trying.png',myfile)
'''