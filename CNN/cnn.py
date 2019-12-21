# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 19:11:37 2019

@author: DelowaR
"""

#part - 1 Building the CNN

#impirt the keras libraries and packages
import tensorflow.keras as keras
from keras.models import Sequential #initialized the nural network
from keras.layers import Convolution2D #create cnn for 2d here image is 2d ,but video is 3d
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense #layer fully connected ann 

#initialization the CNN
classifier=Sequential()

#step-1 Convolution
classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))#32 filte ,3 row and column
#input_shape 64 x 64 image 3 no of channel

#Step-2 - pooling  #reduce array size  
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolution layer
classifier.add(Convolution2D(32,3,3,activation='relu'))#32 filte ,3 row and column
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step-3 Flatten #covrt 2d array to 1d
classifier.add(Flatten())
 
#Step-4 full connection
#adding hidden layer
classifier.add(Dense(output_dim=128,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))
#sigmoid has binary outcome
#output_dim no of node

#Compiling the CNN
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
#Part -2 Fitting the CNN to the images
#E:\Machine_Learning_A_Z\CNN\dataset
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'E:/Machine_Learning_A_Z/CNN/dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'E:/Machine_Learning_A_Z/CNN/dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)


#steps_per_epoch total traing data
#validation_steps total test data


