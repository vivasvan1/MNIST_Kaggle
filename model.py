'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import csv

x_train = []
y_train = []
with open('./digit-recognizer/train.csv') as train_file:
    csv_reader = csv.reader(train_file, delimiter=',')
    i = True
    for row in csv_reader:
        if i ==True:
            i = False
        else:
            y_train.append(int(row[0]))
            x_train.append(list(map(int,row[1:])))
    # print(x_train)
    # print(y_train)

x_test = []
with open('./digit-recognizer/test.csv') as test_file:
    csv_reader = csv.reader(test_file, delimiter=',')
    i = True
    for row in csv_reader:
        if i ==True:
            i = False
        else:
            x_test.append(list(map(int,row)))
    # print(x_test)
    # print(y_test)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
# y_test = np.array(y_test)

print(x_train.shape)

batch_size = 128
num_classes = 10
epochs = 20

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
# x_train = 
# (x_train, y_train), (x_test, y_test) = mnist.load_data()



if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adamax(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

model.save_weights('my_model_weights.h5')
score = model.predict_classes(x_test)
print(score)
with open('sumbision.csv','w') as fil:
    writer = csv.writer(fil, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['ImageId','Label'])
    counter = 1
    for i in score:
        writer.writerow([counter,i])
        counter +=1

# print('Test loss:', score[0])
# print('Test accuracy:', score[1])