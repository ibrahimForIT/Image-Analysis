import numpy as np
import pandas as pd
import os

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout , conv2D
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.metrics import categorical_crossentropy
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import *
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD

from matplotlib import pyplot as plt



import warnings
warnings.filterwarnings('ignore')


print('Tensorflow version = {}'.format(tf.__version__))
print('Keras version = {}'.format(keras.__version__))


NORMALIZER = 1./255
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_TEST = 32 
BATCH_SIZE_VALID = 16
INPUT_DIM = (64, 64, 1)

print(INPUT_DIM[0:2])

# Build the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(64, (4, 4), input_shape=(64,64,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(512, (3, 3))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(512, (2, 2))) 
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    
    model.add(Flatten()) 
    # Dropouts help reduce overfitting by randomly turning neurons off during training.
    # Here we say randomly turn off 50% of neurons.
    model.add(Dropout(0.5))
    model.add(Dense(128)) 
    model.add(Activation('relu'))

    model.add(Dense(1)) 
    model.add(Activation('sigmoid'))
    

    # Define an optimizer for the model
    #opt = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #opt = SGD(lr=0.01, decay=1e-6, momentum=0.85, nesterov=True)
    #opt = RMSprop(lr=0.001, rho=0.8, epsilon=None, decay=0.0)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Deffining the Training and Testing Datasets
train_datagen = ImageDataGenerator(rescale= NORMALIZER,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True)


test_datagen = ImageDataGenerator(rescale=NORMALIZER)
                                
training_set = train_datagen.flow_from_directory(
'./data/train',
target_size=INPUT_DIM[0:2],
batch_size=BATCH_SIZE_TRAIN,
class_mode='binary',
color_mode='grayscale',
shuffle=True
)

test_set = test_datagen.flow_from_directory(
'./data/test',
target_size=INPUT_DIM[0:2],
batch_size=BATCH_SIZE_TEST,
class_mode='binary',
color_mode='grayscale',
shuffle=True
)


model = create_model()
model.summary()

lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=2, mode='max')


Model = model.fit(
training_set,
steps_per_epoch=len(training_set),
epochs=3,
validation_data=test_set,
validation_steps=len(test_set),
verbose = 2 )

model.save('my_model10.h5')

plt.plot(Model.history['loss'])
plt.plot(Model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss10.png')
plt.close()

# Accuracy plotting
plt.plot(Model.history['accuracy'])
plt.plot(Model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy10.png')