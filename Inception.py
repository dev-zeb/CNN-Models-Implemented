import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import AvgPool2D, Dropout, Conv2D, concatenate, Dense, Flatten, Input, MaxPool2D
from keras.models import Model

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

# Define a func. for the inception block
# where 'x' is the input_shape and f is the list of 6 input_shapes for the conv. layers
def inception_block(x, filter_list, name=[]):
  # Unpacking the filters from the list
  filters_1x1 = filter_list[0]
  filters_3x3_reduce = filter_list[1]
  filters_3x3 = filter_list[2]
  filters_5x5_reduce = filter_list[3]
  filters_5x5 = filter_list[4]
  filters_pool_proj = filter_list[5]

  # Define the 1st tower with 1 conv. layer
  t1 = Conv2D(filters_1x1, (1, 1), activation='relu', name=name[0])(x)

  # Define the 2nd tower with 2 conv. layer
  t2 = Conv2D(filters_3x3_reduce, (1, 1), activation='relu', name=name[1])(x)
  t2 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', name=name[2])(t2)
  
  # Define the 3rd tower with 2 conv. layer
  t3 = Conv2D(filters_5x5_reduce, (1, 1), activation='relu', name=name[3])(x)
  t3 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', name=name[4])(t3)
  
  # Define the 4th tower with 1 max pool & 1 conv. layer
  t4 = MaxPool2D(3, (1, 1), padding='same')(x)
  t4 = Conv2D(filters_pool_proj, (1, 1), activation='relu', name=name[5])(t4)

  # Finally concate all the towers
  output = concatenate([t1, t2, t3, t4], name=name[6])

  # Return the output
  return output

# Define a func. for the GoogLeNet model
def googlenet(input_shape, n_classes):
  input = input_shape  # (224 x 224 x 3)
  
  # (64 x 7 x 7)
  x = Conv2D(64, (7, 7), strides=2, padding='same', activation='relu', name="conv2d_64x7x7_s_2")(input)
  x = MaxPool2D((3, 3), strides=2, padding='same')(x)  # (3 x 3)/s(2)
  
  x = Conv2D(64, (1, 1), activation='relu', name="conv2d_64x1x1")(x)
  x = Conv2D(192, (3, 3), padding='same', activation='relu', name="conv2d_192x3x3")(x)
  x = MaxPool2D((3, 3), strides=2)(x)
  
  x = inception_block(x, [64, 96, 128, 16, 32, 32], name=["3a_64x1x1", "reduce_3a_96x1x1", "3a_128x3x3", "reduce_3a_16x1x1", "3a_32x5x5", "pool_proj_3a_32x1x1", "inception_3a"])
  x = inception_block(x, [128, 128, 192, 32, 96, 64], name=["3b_128x1x1", "reduce_3b_128x1x1", "3b_192x3x3", "reduce_3b_32x1x1", "3b_96x5x5", "pool_proj_3b_64x1x1", "inception_3b"])
  x = MaxPool2D((3, 3), strides=2, padding='same')(x)
  
  x = inception_block(x, [192, 96, 208, 16, 48, 64], name=["4a_192x1x1", "reduce_4a_96x1x1", "4a_208x3x3", "reduce_4a_16x1x1", "4a_48x5x5", "pool_proj_4a_64x1x1", "inception_4a"])
  x = inception_block(x, [160, 112, 224, 24, 64, 64], name=["4b_160x1x1", "reduce_4b_112x1x1", "4b_224x3x3", "reduce_4b_24x1x1", "4b_64x5x5", "pool_proj_4b_64x1x1", "inception_4b"])
  x = inception_block(x, [128, 128, 256, 24, 64, 64], name=["4c_128x1x1", "reduce_4c_128x1x1", "4c_256x3x3", "reduce_4c_24x1x1", "4c_64x5x5", "pool_proj_4c_64x1x1", "inception_4c"])
  x = inception_block(x, [112, 144, 288, 32, 64, 64], name=["4d_112x1x1", "reduce_4d_144x1x1", "4d_288x3x3", "reduce_4d_32x1x1", "4d_64x5x5", "pool_proj_4d_64x1x1", "inception_4d"])
  x = inception_block(x, [256, 160, 320, 32, 128, 128], name=["4e_256x1x1", "reduce_4e_160x1x1", "4e_320x3x3", "reduce_4e_32x1x1", "4e_128x5x5", "pool_proj_4e_128x1x1", "inception_4e"])
  x = MaxPool2D((3, 3), strides=2, padding='same')(x)

  x = inception_block(x, [256, 160, 320, 32, 128, 128], name=["5a_256x1x1", "reduce_5a_160x1x1", "5a_320x3x3", "reduce_5a_32x1x1", "5a_128x5x5", "pool_proj_5a_128x1x1", "inception_5a"])
  x = inception_block(x, [384, 192, 384, 48, 128, 128], name=["5b_384x1x1", "reduce_5b_192x1x1", "5b_384x3x3", "reduce_5b_48x1x1", "5b_128x5x5", "pool_proj_5b_128x1x1", "inception_5b"])
  
  x = AvgPool2D((7, 7), strides=1)(x)
  x = Dropout(0.4)(x)
  
  x = Flatten()(x)
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model

input_shape = Input(shape=(224, 224, 3))
num_classes = 5

inception = googlenet(input_shape, num_classes)
inception.summary()
inception.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# define some hyper-parameters
num_classes = 5
batch_size = 5
num_epochs = 5

# create a data generator
datagen = ImageDataGenerator()

# load and iterate training dataset
train_set = datagen.flow_from_directory('D:/A2/Dataset/train', target_size=(224, 224), class_mode='categorical', batch_size=batch_size)
# load and iterate validation dataset
val_set = datagen.flow_from_directory('D:/A2/Dataset/validation', target_size=(224, 224), class_mode='categorical', batch_size=batch_size)
# load and iterate test dataset
test_set = datagen.flow_from_directory('D:/A2/Dataset/test', target_size=(224, 224), class_mode='categorical', batch_size=batch_size)

# calculate the following values
steps_per_epoch = len(train_set)//batch_size
validation_steps = len(val_set)//batch_size

# fit model
inception.fit(train_set, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val_set, validation_steps=validation_steps)

# evaluate model
loss = inception.evaluate(test_set, steps=24)

# make a prediction
yhat = inception.predict(val_set, steps=24)

print("train_set.next(): ", type(train_set))
# confirm the iterator works
batchX, batchy = train_set.next()

print("type(loss): ", type(loss))
print("len(loss): ", len(loss))
print("loss: ", loss)
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))
