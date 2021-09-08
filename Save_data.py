import pickle
import random
import os
import numpy as np


DATA_DIR = "D:/A2/Pet_Images"
CATEGORIES = ["Dog", "Cat"]

train_set = []

def create_train_dataset():
  for category in CATEGORIES:
    path = os.path.join(DATA_DIR, category)  # Path to cats or dogs
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
      try:
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        train_set.append([img_array, class_num])
      except Exception as e:
        pass


create_train_dataset()

'''
print(len(train_set))

random.shuffle(train_set)
for sample in train_set[:10]:
  print(sample[1])
'''
X = []
y = []
IMG_SIZE = 150

for features, label in train_set:
  X.append(features)
  y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)

# Saving the data
pickle_out = open("X.npy", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()


pickle_out = open("y.npy", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

print("Data saved")
