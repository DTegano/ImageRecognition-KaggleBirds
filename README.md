# Image Recognition - Kaggle Birds

This repository demonstrates a Convolutional Neural Network used on Kaggle's 180 Bird Species Data Set. Within this data, there are 24,507 training images for the 180 different bird species, along with 900 validation and test images. This project was done using Google Colaboratory. 

# Libraries

My first step was to import all of the necessary libraries needed to complete this project. I utilized keras and tensorflow for the majority of my libraries. As you may notice, I set my seed to a particular value so that I can repeat randomization techniques and get a similar result.

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

# Exploring the Data

Once all of my directories are in place, I ran a few commands to test out whether the directories were successful in their uploading. While I randomly ran a few bird species to ensure success, the below only shows the Albatross species. As we can see, there are 180 bird species in the train, validation, and test directory. For the Albatross species, there are 100 images in the training directory, and 5 images in each of the validation and test directories. This would likely indicate that the dataset contains 18,000 train images (180 species 100 images), 900 validation and testing images (180 x 5). However, as we'll see shortly, the training images doesn't appear consistent with the total # of images in the training set.

```
print(len(os.listdir(train_dir)))
print(len(os.listdir(train_ALBATROSS_dir)))
print(len(os.listdir(validation_dir)))
print(len(os.listdir(validation_ALBATROSS_dir)))
print(len(os.listdir(test_dir)))
print(len(os.listdir(test_ALBATROSS_dir)))
```

<img src = "https://user-images.githubusercontent.com/39016197/92836499-b591d280-f399-11ea-897d-ae395ffca092.png" width = 100 height = 150>

To check these potential inconsistencies, I checked the length of 2 other species. My suspicion is confirmed when we see that the 3 species below all have a different # of training images (100, 166, and 180):

```
# Checking for inconsistent training images among species
print(len(os.listdir(train_ALBATROSS_dir)))
print(len(os.listdir(validation_ALBATROSS_dir)))
print(len(os.listdir(test_ALBATROSS_dir)))
print(len(os.listdir(train_ALEXANDRINEPARAKEET_dir)))
print(len(os.listdir(validation_ALEXANDRINEPARAKEET_dir)))
print(len(os.listdir(test_ALEXANDRINEPARAKEET_dir)))
print(len(os.listdir(train_AMERICANAVOCET_dir)))
print(len(os.listdir(validation_AMERICANAVOCET_dir)))
print(len(os.listdir(test_AMERICANAVOCET_dir)))
```
<img src = "https://user-images.githubusercontent.com/39016197/92838839-82047780-f39c-11ea-88bd-82beaafdc69b.png" width = 100 height = 200>

# Data Augmentation

Next, I'll use the data augmentation techniques below to replace the existing data with eventual image copies that have been rotated, zoomed-in on, etc. This will only be applied to the training and validation directories. The testing data will still needs to be rescaled so this will be handled separately without augmentation. As we can see below, the data set contains 24,507 training images, 900 test and 900 validation images. I've modified the batch sizes so that the epoch steps will only be 100 for the training data, and 30 steps for the validation/test sets (30 batches * 30 steps per epoch = 900). Since 24,507 training images is hardly a divisible number, I chose to run with a batch size of 245.

```
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255) 

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=246,
    class_mode='categorical')

validataion_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=30,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=30,
    class_mode='categorical')
```
<img src = "https://user-images.githubusercontent.com/39016197/92839322-1242bc80-f39d-11ea-8735-014b2a3deb08.png" width = 500 height = 100>

Next, to make sure my data augmentation worked correctly, I'll pull up an image of a random picture - in this case, it was the Rosy-Faced Loved Bird. Next, I'll compare the original photo to the augmented photos:

Here is the original image: <p></p>
<img src = "https://user-images.githubusercontent.com/39016197/92841932-2805b100-f3a0-11ea-9616-a3ff7f48d89a.png" width = 280 height = 280>


Here are the augmented photos:

```
img = image.load_img(os.path.join(train_ROSYFACEDLOVEBIRD_dir, os.listdir(train_ROSYFACEDLOVEBIRD_dir)[1]), target_size=(224,224))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0 
for batch in train_datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()
```
<img src = "https://user-images.githubusercontent.com/39016197/92842553-ef1a0c00-f3a0-11ea-9da8-e918730c3943.png" width = 300 height = 600>
<img src = "https://user-images.githubusercontent.com/39016197/92842741-2ab4d600-f3a1-11ea-92be-5f6b0232d069.png" width = 300 height = 600>

# Convolutional Neural Network

Now that my data is set up for machine learning, I'll load in my transfer learning model. In the library list above, I chose to load in the 'xception' model from keras. Since this is a pre-trained data set, using transfer learning will save my model a bunch of time on learning simple patterns that the xception model has already been trained using the ImageNet (which is the weights I'll be loading in). I'll also print a summary command to see the architecture of this model. I'll note that since the architecture of the model is long, I'll only show the last bit of the summary:

```
backend.clear_session()
base_model = Xception (weights = 'imagenet', # using the weights trained on imagenet
                  include_top = False,
                  input_shape = (224, 224, 3))

print('Xception Model', base_model.summary())
```
<img src = "https://user-images.githubusercontent.com/39016197/92843538-158c7700-f3a2-11ea-8e4b-720f98afefa8.png" width = 800 height = 800>
