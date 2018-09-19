import mnist
from scipy.misc import imsave
import numpy as np


train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()
test_labels = mnist.test_labels()


#imsave('test.png', train_images[0])
global_count = 0
count = (np.zeros((10))).astype(dtype='int32')
for digit in train_labels:
    path = 'mnist_dataset//training_set//'
    if digit == 0:
        file_name =  path + '0//zero_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 1:
        file_name = path + '1//one_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 2:
        file_name = path + '2//two_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 3:
        file_name = path + '3//three_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 4:
        file_name = path + '4//four_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 5:
        file_name = path + '5//five_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 6:
        file_name = path + '6//six_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 7:
        file_name = path + '7//seven_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 8:
        file_name = path + '8//eight_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 9:
        file_name = path + '9//nine_' + str(count[digit]) + '.png'
        imsave(file_name, train_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1



global_count = 0
count = (np.zeros((10))).astype(dtype='int32')
for digit in test_labels:
    path = 'mnist_dataset//test_set//'
    if digit == 0:
        file_name =  path + '0//zero_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 1:
        file_name = path + '1//one_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 2:
        file_name = path + '2//two_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 3:
        file_name = path + '3//three_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 4:
        file_name = path + '4//four_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 5:
        file_name = path + '5//five_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 6:
        file_name = path + '6//six_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 7:
        file_name = path + '7//seven_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 8:
        file_name = path + '8//eight_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1

    elif digit == 9:
        file_name = path + '9//nine_' + str(count[digit]) + '.png'
        imsave(file_name, test_images[global_count])
        global_count = global_count + 1
        count[digit] = count[digit] + 1
