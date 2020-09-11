# Image Recognition- Kaggle Birds

This repository demonstrates a Convolutional Neural Network used on Kaggle's 180 Bird Species Data Set. Within this data, there are 24,507 training images for the 180 different bird species, along with 900 validation and test images. This project was done using Google Colaboratory. 

# Libraries

```
%tensorflow_version 2.x
import tensorflow as tf
from tensorflow.keras import models, backend, layers, optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D 
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend, models, layers, optimizers, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing import image # Used to view data augmented images
from tensorflow.keras.applications import Xception # Importing the xception model for transfer learning
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Library for data augmentation
import numpy as np
import pandas as pd
from google.colab import files
from google.colab import drive
from IPython.display import display 
from PIL import Image
import os, shutil # Library for navigating files
np.random.seed(18)
```
# Importing and setting up the Data
```
drive.mount('/content/gdrive')
data_dir = '/content/gdrive/My Drive/KaggleBirds/180'
```

With a data set this large, I found it was best to upload the images to my google drive and import the data from my Google Drive once mounted. Next, I need to set up separate directories from my training, validation, and test images.

```
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')
```

The below was probably the biggest nuisance of this project. For each directory, I set up an individual directory for each bird species:

```
train_ALBATROSS_dir = os.path.join(train_dir, 'ALBATROSS')
train_ALEXANDRINEPARAKEET_dir = os.path.join(train_dir, 'ALEXANDRINE PARAKEET')
train_AMERICANAVOCET_dir = os.path.join(train_dir, 'AMERICAN AVOCET')
train_AMERICANBITTERN_dir = os.path.join(train_dir, 'AMERICAN BITTERN')
train_AMERICANCOOT_dir = os.path.join(train_dir, 'AMERICAN COOT')
train_AMERICANGOLDFINCH_dir = os.path.join(train_dir, 'AMERICAN GOLDFINCH')
train_AMERICANKESTREL_dir = os.path.join(train_dir, 'AMERICAN KESTREL')
train_AMERICANPIPIT_dir = os.path.join(train_dir, 'AMERICAN PIPIT')
train_AMERICANREDSTART_dir = os.path.join(train_dir, 'AMERICAN REDSTART')
train_ANHINGA_dir = os.path.join(train_dir, 'ANHINGA')
train_ANNASHUMMINGBIRD_dir = os.path.join(train_dir, 'ANNAS HUMMINGBIRD')
train_ANTBIRD_dir = os.path.join(train_dir, 'ANTBIRD')
train_ARARIPEMANAKIN_dir = os.path.join(train_dir, 'ARARIPE MANAKIN')
train_BALDEAGLE_dir = os.path.join(train_dir, 'BALD EAGLE')
...

validation_ALBATROSS_dir = os.path.join(validation_dir, 'ALBATROSS')
validation_ALEXANDRINEPARAKEET_dir = os.path.join(validation_dir, 'ALEXANDRINE PARAKEET')
validation_AMERICANAVOCET_dir = os.path.join(validation_dir, 'AMERICAN AVOCET')
validation_AMERICANBITTERN_dir = os.path.join(validation_dir, 'AMERICAN BITTERN')
validation_AMERICANCOOT_dir = os.path.join(validation_dir, 'AMERICAN COOT')
validation_AMERICANGOLDFINCH_dir = os.path.join(validation_dir, 'AMERICAN GOLDFINCH')
validation_AMERICANKESTREL_dir = os.path.join(validation_dir, 'AMERICAN KESTREL')
validation_AMERICANPIPIT_dir = os.path.join(validation_dir, 'AMERICAN PIPIT')
validation_AMERICANREDSTART_dir = os.path.join(validation_dir, 'AMERICAN REDSTART')
validation_ANHINGA_dir = os.path.join(validation_dir, 'ANHINGA')
validation_ANNASHUMMINGBIRD_dir = os.path.join(validation_dir, 'ANNAS HUMMINGBIRD')
validation_ANTBIRD_dir = os.path.join(validation_dir, 'ANTBIRD')
validation_ARARIPEMANAKIN_dir = os.path.join(validation_dir, 'ARARIPE MANAKIN')
validation_BALDEAGLE_dir = os.path.join(validation_dir, 'BALD EAGLE')
...
```

# Convolutional Neural Network
```
print(len(os.listdir(train_dir)))
print(len(os.listdir(train_ALBATROSS_dir)))
print(len(os.listdir(validation_dir)))
print(len(os.listdir(validation_ALBATROSS_dir)))
print(len(os.listdir(test_dir)))
print(len(os.listdir(test_ALBATROSS_dir)))
```

<img src = "https://user-images.githubusercontent.com/39016197/92836499-b591d280-f399-11ea-897d-ae395ffca092.png" width = 100 height = 150>
