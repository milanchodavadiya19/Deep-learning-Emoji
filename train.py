import numpy as np
import cv2

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator


# All layers after import

# two directory
train_dir = 'data/train'
val_dir = 'data/test'


###
#the script sets up two directories for the training and validation data. 
#The data is loaded using the ImageDataGenerator class from Keras, which is used to pre-process the image data. 
#The data is rescaled so that all pixel values are in the range [0,1].
###


# Image from data with scale
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)


###
#The script then sets up two generators for the training and validation data, using the flow_from_directory method of the ImageDataGenerator class. 
#This method loads the images from the specified directories, resizes them to 48x48 pixels, and divides them into batches of 64 images.
###
# tareget size 48*48 pixels
# batch size 

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode="categorical")

validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


# create model and prepraring for training
# emotion_model variable creating
# keras we have two option - Sequential and functional
# add layers 
# 2D Convunational neural networks layer-- input 32, width and height 3*3, input_shape = size of pictures, 
# MaxPooling2D 
# Dense layer -- 1024 neurons convert to 7 neurons
#####
#The script then creates an instance of the Sequential class from Keras, which represents a linear stack of layers in the neural network. 
#The script adds several layers to the network, including Conv2D, MaxPooling2D, Flatten, and Dense layers. 
#The Conv2D layer performs convolution operations on the image data, while the MaxPooling2D layer performs pooling operations to reduce the size of the feature maps. 
#The Flatten layer converts the feature maps into a 1D array, and the Dense layer implements a fully connected neural network. 
#Dropout layers are used to prevent overfitting during training.
#####

emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(48,48,1)))
emotion_model.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))



# compiler model -------- Compilation ---------------
# loss- degree of error 
# Adam standrad optimizer used here 
# track accuracy in metrics
###
#Finally, the script compiles the model using the compile method. 
#This method sets the loss function to be used during training (categorical cross-entropy), the optimizer to be used (Adam), and the metrics to track (accuracy). 
#The script then trains the model using the fit method, specifying the number of steps per epoch and the number of epochs to train for. 
#Finally, the model's weights are saved to a file using the save_weights method.
###

emotion_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001, decay=1e-6), metrics=['accuracy'])

# train model
emotion_model_info = emotion_model.fit(
        train_generator,
        steps_per_epoch=28709 // 64,
        epochs=50, # 50
        validation_data=validation_generator,
        validation_steps=7178 // 64)

emotion_model.save_weights('model.h5')                          # save weights
