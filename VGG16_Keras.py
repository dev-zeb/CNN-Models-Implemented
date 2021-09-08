
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
from keras.models import Model

class VGG16:
  def __init__(self, num_classes=1000):
    # define input layer
    self.input_layer = Input((224, 224, 3), name='input_layer')
    
    #                                                         ----------------------------------------------------------------------------------------------------------------------
    # Block 1                                                /                                                                                                                      \ 
    # No. of input features: 224 x 224 x 3                  /                           (total input features) * [# of filters in cur. layer] + bias                                 \  
    # 2 conv. layers having 64 filters of size 3x3         / (# of filters in prev layer*size_of_filters in prev layer) * [# of filters in cur. layer] + {# of filters in cur. layer} \
    self.x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1_64x3x3')(self.input_layer) # Learnable Param. #: (3 * 3*3) * [64] + {64}
    self.x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2_64x3x3')(self.x) # Learnable Param. #: 64* 3*3 *64 + 64
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_1_2x2_s_2')(self.x) # Learnable Param. #: 0
    # No. of output features: 112*112*3 i.e img_size*img_size*filter_size

    # Block 2
    # No. of input features: 112*112*3 i.e. output features of the prev. block
    self.x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1_128x3x3')(self.x) # Learnable Param. #: 64* 3*3 *128 +128
    self.x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2_128x3x3')(self.x) # Learnable Param. #: 128* 3*3 *128 +128
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_1_2x2_s_2')(self.x) # Learnable Param. #: 0
    # No. of output features: 56*56*3 i.e img_size*img_size*filter_size
    
    # Block 3
    # No. of input features: 56*56*3 i.e. output features of the prev. block
    self.x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1_256x3x3')(self.x) # Learnable Param. #: 128* 3*3 *256 +256
    self.x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2_256x3x3')(self.x) # Learnable Param. #: 256* 3*3 *256 +256
    self.x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3_256x3x3')(self.x) # Learnable Param. #: 256* 3*3 *256 +256
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_1_2x2_s_2')(self.x) # Learnable Param. #: 0
    # No. of output features: 28*28*3 i.e img_size*img_size*filter_size

    # Block 4
    # No. of input features: 28*28*3 i.e. output features of the prev. block
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1_512x3x3')(self.x) # Learnable Param. #: 256* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2_512x3x3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3_512x3x3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4_1_2x2_s_2')(self.x) # Learnable Param. #: 0
    # No. of output features: 14*14*3 i.e img_size*img_size*filter_size

    # Block 4
    # No. of input features: 14*14*3 i.e. output features of the prev. block
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1_512x3x3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2_512x3x3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3_512x3x3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5_1_2x2_s_2')(self.x) # Learnable Param. #: 0
    #                      (# of output features)
    #          (img_size_pixel * img_size_pixel * # of filters)
    # No. of output features: (7 * 7 * 512) || initially, img_size_pixel = 224 x 224; after 5 maxpooling of size=(2, 2); img_size_pixel = 224 / (2 ** 5) = 7
    
    # Block 5
    # No. of input features: 7 * 7 * 512 i.e. output features of the prev. block
    self.x = Flatten(name='flatten')(self.x) # Flatten # of features from 7 * 7 * 512 to 25088; which is the # of inputs of the next layer
    self.x = Dense(4096, name='fc1', activation='relu')(self.x) # Learnable Param. #: (25088) * [4096] + {4096}; i.e. (# of input features) * [neoron # of cur. layer] + {neoron # of cur. layer}
    self.x = Dense(4096, name='fc2', activation='relu')(self.x) # Learnable Param. #: (4096) * [4096] + {4096}; i.e. (neoron # of prev. layer) * [neoron # of cur. layer] + {neoron # of cur. layer}
    self.x = Dense(num_classes, name='fc3', activation='softmax')(self.x) # Learnable Param. #: (4096) * [1000] + {1000}; i.e. (neoron # of prev. layer) * [neoron # of cur. layer] + {neoron # of cur. layer}
    # No. of output features: num_classes

    # Build model
    self.model = Model(self.input_layer, self.x, name='VGG16')


# First define some hyper-parameters
num_classes = 5 # Based on the dataset used
batch_size = 5
num_epochs = 25

# Create an instance of the 'VGG16' class with the number of classes
vgg16 = VGG16(num_classes)
vgg16.model.summary()
vgg16.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create a data generator
datagen = ImageDataGenerator()

import tflite_model_maker
# Load the training dataset
train_set = datagen.flow_from_directory('D:/A2/Dataset/train', target_size=(224, 224), class_mode='categorical', batch_size=batch_size)
# Load the validation dataset
val_set = datagen.flow_from_directory('D:/A2/Dataset/validation', target_size=(224, 224), class_mode='categorical', batch_size=batch_size)
# Load the test dataset
test_set = datagen.flow_from_directory('D:/A2/Dataset/test', target_size=(224, 224), class_mode='categorical', batch_size=batch_size)

# Calculate the following values
steps_per_epoch = len(train_set)//batch_size
validation_steps = len(val_set)//batch_size

# Fit the model
vgg16.model.fit(train_set, epochs=num_epochs, steps_per_epoch=16, validation_data=val_set, validation_steps=8)

# Evaluate the model
loss, acc = vgg16.model.evaluate(test_set, steps=24)

# Get results on validation set
yhat = vgg16.model.predict(val_set, steps=24)

# confirm the iterator works
batchX, batchy = train_set.next()

print("loss: ", loss, ", acc: ", acc)
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))


'''
#-----------------------------------------------------------------------
batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    'D://A2//Dataset//train',  # this is the target directory
    target_size=(224, 224),  # all images will be resized to 150x150
    batch_size=batch_size,
    class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
    'D://A2//Dataset//test',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='binary')

vgg16.model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=800 // batch_size
)

----------------Lemon
#from google import drive

#drive.mount("/content/drive", force_remount=True)
imagegen = ImageDataGenerator(rescale=1/255.)
test = imagegen.flow_from_directory("D://A2//Dataset//test", class_mode="categorical", shuffle=False, batch_size=5, target_size=(224,224))
train = imagegen.flow_from_directory("D://A2//Dataset//train", class_mode="categorical", shuffle=False, batch_size=5, target_size=(224,224))
print(train.image_shape, test.image_shape)

EPOCH_STEPS = 1
EPOCHS = 10
train_history = vgg16.model.fit(
    train,
    steps_per_epoch=EPOCH_STEPS,
    epochs=EPOCHS,
    validation_data=test,#callbacks=[tensorboard_callback],
    shuffle=True
)
'''
'''
class VGG16:
  def __init__(self, num_classes=1000):
    # define input layer
    self.input_layer = Input([224, 224, 3], name='input_layer')
    
    #                                                         ----------------------------------------------------------------------------------------------------------------------
    # Block 1                                                /                                                                                                                      \ 
    # No. of input features: 224 x 224 x 3                  /                           (total input features) * [# of filters in cur. layer] + bias                                 \  
    # 2 conv. layers having 64 filters of size 3x3         / (# of filters in prev layer*size_of_filters in prev layer) * [# of filters in cur. layer] + {# of filters in cur. layer} \
    self.x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_1')(self.input_layer) # Learnable Param. #: (3 * 3*3) * [64] + {64}
    self.x = Conv2D(64, (3, 3), padding='same', activation='relu', name='conv1_2')(self.x) # Learnable Param. #: 64* 3*3 *64 + 64
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool1_1')(self.x) # Learnable Param. #: 0
    # No. of output features: 112*112*3 i.e img_size*img_size*filter_size

    # Block 2
    # No. of input features: 112*112*3 i.e. output features of the prev. block
    self.x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_1')(self.x) # Learnable Param. #: 64* 3*3 *128 +128
    self.x = Conv2D(128, (3, 3), padding='same', activation='relu', name='conv2_2')(self.x) # Learnable Param. #: 128* 3*3 *128 +128
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2_1')(self.x) # Learnable Param. #: 0
    # No. of output features: 56*56*3 i.e img_size*img_size*filter_size
    
    # Block 3
    # No. of input features: 56*56*3 i.e. output features of the prev. block
    self.x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_1')(self.x) # Learnable Param. #: 128* 3*3 *256 +256
    self.x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_2')(self.x) # Learnable Param. #: 256* 3*3 *256 +256
    self.x = Conv2D(256, (3, 3), padding='same', activation='relu', name='conv3_3')(self.x) # Learnable Param. #: 256* 3*3 *256 +256
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool3_1')(self.x) # Learnable Param. #: 0
    # No. of output features: 28*28*3 i.e img_size*img_size*filter_size

    # Block 4
    # No. of input features: 28*28*3 i.e. output features of the prev. block
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_1')(self.x) # Learnable Param. #: 256* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_2')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv4_3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool4_1')(self.x) # Learnable Param. #: 0
    # No. of output features: 14*14*3 i.e img_size*img_size*filter_size

    # Block 4
    # No. of input features: 14*14*3 i.e. output features of the prev. block
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_1')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_2')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = Conv2D(512, (3, 3), padding='same', activation='relu', name='conv5_3')(self.x) # Learnable Param. #: 512* 3*3 *512 +512
    self.x = MaxPooling2D((2, 2), strides=(2, 2), name='pool5_1')(self.x) # Learnable Param. #: 0
    #                      (# of output features)
    #          (img_size_pixel * img_size_pixel * # of filters)
    # No. of output features: (7 * 7 * 512) || initially, img_size_pixel = 224 x 224; after 5 maxpooling of size=(2, 2); img_size_pixel = 224 / (2 ** 5) = 7
    
    # Block 5
    # No. of input features: 7 * 7 * 512 i.e. output features of the prev. block
    self.x = Flatten(name='flatten')(self.x) # Flatten # of features from 7 * 7 * 512 to 25088; which is the # of inputs of the next layer
    self.x = Dense(4096, name='fc1', activation='relu')(self.x) # Learnable Param. #: (25088) * [4096] + {4096}; i.e. (# of input features) * [neoron # of cur. layer] + {neoron # of cur. layer}
    self.x = Dense(4096, name='fc2', activation='relu')(self.x) # Learnable Param. #: (4096) * [4096] + {4096}; i.e. (neoron # of prev. layer) * [neoron # of cur. layer] + {neoron # of cur. layer}
    self.x = Dense(num_classes, name='fc3', activation='softmax')(self.x) # Learnable Param. #: (4096) * [1000] + {1000}; i.e. (neoron # of prev. layer) * [neoron # of cur. layer] + {neoron # of cur. layer}
    # No. of output features: num_classes

    # Build model
    self.model = Model(self.input_layer, self.x, name='VGG16')


# First define some hyper-parameters
num_classes = 1000
batch_size = 1
num_epochs = 1

# Create an instance of the 'VGG16' class
vgg16 = VGG16(num_classes)
vgg16.model.summary()
#vgg16.model.save('VGG16.h5')

path = 'D:\Study\10th Semester(4th Year)\Machine Learning\FINAL\ML Assignments\datasets'
data_path = os.listdir(path)
room_types = os.listdir(path)

print(room_types)
print("Types of rooms found: ", len(data_path))

# Define an empty list
rooms = []

for item in room_types:
  #print(item)
  # Avoiding any hidden files or folders
  if item.startswith('.'):
    continue
  # Get all the file names
  all_rooms = os.listdir(path + '/' + item)
  for room in all_rooms:
    # Add the room to the list named rooms
    rooms.append((item, str(path + '/' + item + '/' + room)))
    #print(len(rooms))

# Following lines of code checks if it's stored or not
rooms_df = pd.DataFrame(data=rooms, columns=['room_type', 'image'])
room_count = rooms_df['room_type'].value_counts()
print(rooms_df.head())
print("Total # of rooms: ", len(rooms_df))
print("Rooms in each category: \n", room_count)

import cv2
# Define the image pixel size
img_size = 224

# Define 2 lists to store the images and their corresponding labels
images = []
labels = []

# Run loop over the room_types list
for i in room_types:
  # Store the full path in 'data_path'
  data_path = path + '/' + str(i)
  # Store all the images inside the 'data_path' directory in the 'filenames' list
  filenames = [i for i in os.listdir(data_path)]
  
  # Iterate over each image
  for f in filenames:
    # Store the image in 'img' variable
    img = cv2.imread(data_path+'/'+f)
    # Resize the image to the desired images size using 'img_size'
    img = cv2.resize(img, (img_size, img_size))
    # Add the resized image to the 'images' list
    images.append(img)
    # Add its corresponding label to the 'labels' list
    labels.append(i)

# Change the 'images' list to a ndarray
print(type(images))
images = np.array(images)
print(type(images))
# Normalize the images stored in 'images' list
images = images.astype('float32') / 255.0
print('X.shape: ', images.shape)

# Import the necessary libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

y = rooms_df['room_type'].values
print('y: ', y)

# Define an instance of LabelEncoder() class
y_labelencoder = LabelEncoder()
# Transform categorical labels to numeric labels of the list 'y'
y = y_labelencoder.fit_transform(y)
print(y)
print('before', y.shape)

# Change the shape of y
y = y.reshape(-1, 1)
print('After', y.shape)
# Define an instance of OneHotEncoder() class
#ohe = OneHotEncoder(categories=[0])
#Y = ohe.fit_transform(y)
#ct = ColumnTransformer([("room_type", OneHotEncoder(),[0])], remainder="passthrough"))
# Transform the scalar output into vector output 
#Y = ct.fit_transform(y)
#Y.shape

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

images, y = shuffle(images, y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, y, test_size=.15, random_state=4)
train_y = np.asarray(train_y).astype('float32').reshape((-1,1))
test_y = np.asarray(test_y).astype('float32').reshape((-1,1))
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# First define some hyper-parameters
num_classes = len(room_types) - 1
batch_size = 1
num_epochs = 1

# Create an instance of the 'VGG16' class
vgg16 = VGG16(num_classes)
vgg16.model.summary()

# Compile the model
vgg16.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model
vgg16.model.fit(train_x, train_y)
'''
