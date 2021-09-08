import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

def conv_bn_rl(x, f, k=1, s=1, p='same'):
  x = Conv2D(f, (k, k), strides=s, padding=p)(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return x

def identity_block(tensor, f):
  x = conv_bn_rl(tensor, f)
  x = conv_bn_rl(x, f, 3)
  x = Conv2D(4*f, (1, 1))(x)
  x = BatchNormalization()(x)

  x = Add()([x, tensor])
  output = ReLU()(x)
  return output

def conv_block(tensor, f, s):
  x = conv_bn_rl(tensor, f)
  x = conv_bn_rl(x, f, 3, s)
  x = Conv2D(4*f, 1)(x)
  x = BatchNormalization()(x)

  shortcut = Conv2D(4*f, 1, strides=s)(tensor)
  shortcut = BatchNormalization()(shortcut)

  x = Add()([x, shortcut])
  output = ReLU()(x)
  return output

def ResNet_block(x, f, r, s=2):
  x = conv_block(x, f, s)
  for _ in range(r-1):
    x = identity_block(x, f)
  return x

def ResNet(input_shape, n_classes):
  input = Input(input_shape)
                      #  f, k, s; f = #of filters, k = kernel size, s = strides
  x = conv_bn_rl(input, 64, 7, 2)
  x = MaxPool2D(3, strides=2, padding='same')(x)

                    #  f, r, s; f = #of filters, r = the #of tym identity_block to call, s = strides
  x = ResNet_block(x, 64, 3, 1)
  x = ResNet_block(x, 128, 4)
  x = ResNet_block(x, 256, 6)
  x = ResNet_block(x, 512, 3)

  x = GlobalAvgPool2D()(x)

  output = Dense(n_classes, activation='softmax')(x)

  model = Model(input, output)
  return model

# Define the input_shape and the number of classes
INPUT_SHAPE = (224, 224, 3)
num_classes = 5

# Build the model and compile it
resnet = ResNet(INPUT_SHAPE, num_classes)
resnet.summary()
resnet.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

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
resnet.fit(train_set, epochs=num_epochs, steps_per_epoch=steps_per_epoch, validation_data=val_set, validation_steps=validation_steps)

# evaluate model
loss = resnet.evaluate(test_set, steps=24)

# make a prediction
yhat = resnet.predict(val_set, steps=24)

# confirm the iterator works
batchX, batchy = train_set.next()

print("type(loss): ", type(loss))
print("len(loss): ", len(loss))
print("loss: ", loss)
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))




"""
What, Why and How things happend are answered in The Glorious Quran.
Not Where and When. As The Honoured Quran is not bounded by Time &
Place, so these two questions are not always answered.

"""


