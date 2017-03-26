import os
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Lambda, SpatialDropout2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

from utils import generator
from utils import random_drop

# load data
samples = []

def make_model_2(input_shape=(72, 320, 3)):
    def resize(img):
        import tensorflow as tf
        return tf.image.resize_images(img, (66, 200))

    model = Sequential()
    # model.add(Flatten(input_shape=input_shape))
    # model.add(Dense(1))

    # resize to 66 200
    # model.add(Cropping2D(cropping=((64,26), (0,0)), input_shape=input_shape))
    model.add(Lambda(resize, input_shape=input_shape))

    # normali
    model.add(Lambda(lambda x: x / (255 / 2.) - 1, input_shape=input_shape))

    # to 24 channel
    model.add(Convolution2D(24, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))

    # to 36 channel
    model.add(Convolution2D(36, 5, 5, border_mode="same", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))

    # to 48 channel
    model.add(Convolution2D(48, 5, 5, border_mode="valid", subsample=(2,2), activation="elu"))
    model.add(SpatialDropout2D(0.2))

    # to 64 channel
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    # to 64 channel twice
    model.add(Convolution2D(64, 3, 3, border_mode="valid", activation="elu"))
    model.add(SpatialDropout2D(0.2))

    model.add(Flatten())
    # model.add(Dropout(0.5))
    model.add(Dense(1164, activation="elu"))
    model.add(Dense(100, activation="elu"))
    model.add(Dense(50, activation="elu"))
    model.add(Dense(10, activation="elu"))
    # model.add(Dropout(0.5))
    model.add(Dense(1))

    model.compile(optimizer=Adam(lr=0.001), loss='mse')

    return model

with open('../CarND-Behavioral-Cloning-P3-Other/data1/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split train and validation data
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

model = make_model_2()
# Print out summary of the model
model.summary()

# Compile model using Adam optimizer and loss computed by mean squared error
# model.compile(loss='mse', optimizer='adam')
model.compile(optimizer=Adam(lr=0.0001), loss='mse')

# random drop image
train_samples = random_drop(samples)
# train
history = model.fit_generator(generator(train_samples, batch_size=64), \
                              samples_per_epoch=int(len(train_samples) / 64) * 64, \
                              validation_data=generator(validation_samples, batch_size=64), \
                              nb_val_samples=len(validation_samples), \
                              nb_epoch=100)

# values = model.fit_generator(generator0(training_data, 64, DATA_PATH), \
#                              samples_per_epoch=int(len(training_data) / 64) * 64,\
#                              nb_epoch=1,\
#                              validation_data=generator1(validation_data, DATA_PATH), nb_val_samples=len(validation_data))


model.save('model/model.3.h5')
exit()

# release memory
import gc;gc.collect()
