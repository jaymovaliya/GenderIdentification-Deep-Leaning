# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.


"""
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(128,128,3), data_format='channels_last' , padding='same'))

classifier.add(MaxPooling2D(pool_size = (2,2) ,padding='same'))

classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))

classifier.add(MaxPooling2D(pool_size = (2,2) ,padding='same'))

classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))

classifier.add(MaxPooling2D(pool_size = (2,2) ,padding='same'))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss= 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(128,128),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=1884,
        epochs=25,
        validation_data=test_set,
        validation_steps=423)


