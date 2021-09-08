
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

VGG16 = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGG_net(nn.Module): # The VGG_net class is inharitating the nn.Module class
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG_net, self).__init__() # __init of the parent method
        self.in_channels = in_channels # Sets the no. of input channels
        # Calling the create_conv_layers func. with the VGG16 list containing all the layers in it
        self.conv_layers = self.create_conv_layers(VGG16)

        # Creating the fully connected layers
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096), # 512 = no. of channels; 7x7 = the images' size after all the MaxPool & Convolutional layers
            nn.ReLU(), # Then a ReLU()
            nn.Linear(4096, 4096), # Then a Linear FCS Layer of 4096 in_channels and out_channels
            nn.ReLU(), # Then another ReLU()
            nn.Linear(4096, num_classes) # Finally another Linear FCS Layer of 4096 in_channels and output layers of the number of classes
        )
    
    def forward(self, x): # Over-riding from the nn.Module parent class. Defines the computation performed at every call.
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fcs(x)

        return x

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == int: # then it's a conv layer
                out_channels = x # no. of out_channels are set to x

                layers +=   [
                                # Creating the stack of convolutional layers
                                nn.Conv2d(  in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=(3, 3), # 3x3 filter
                                            stride=(1, 1),
                                            padding=(1, 1) # To preserve the image's spatial resolution after convolution
                                        ),
                                # Adding an activation function
                                nn.ReLU()
                            ]
                in_channels = x # in_channels for the next layer are set to x 
            elif x == 'M': # then, the Max Pooling layer is added to the 'layers' list
                layers += [
                               nn.MaxPool2d (
                                                kernel_size=(2, 2), # 3x3 filter
                                                stride=(2, 2)
                                            )
                          ]
        # Making a sequence of the layers
        return nn.Sequential(*layers) # *layers is unpacking all the layers inside the 'layers' list

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_channels = 3
num_classes = 1000
learning_rate = 0.001
batch_size = 64
num_epochs = 1

# Load Data and Split that Dataset into train and test Datasets
dataset = datasets.FakeData(size=1500, image_size=(3, 224, 224), num_classes=num_classes, transform=transforms.ToTensor())
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [1150, 350], generator=torch.Generator().manual_seed(7))
'''
print('type-1: ', type(train_dataset))
print('type-2: ', type(test_dataset))
print('size-1: ', len(train_dataset))
print('size-2: ', len(test_dataset))
print('0th: ', train_dataset[0] == test_dataset[0])
'''
# Initialize the Model
model = VGG_net(in_channels=3, num_classes=num_classes).to(device)  # Building the VGG_net model
#print(model)
