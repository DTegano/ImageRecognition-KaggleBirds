# Image Recognition - Kaggle's Bird Data Set

This repository demonstrates a Convolutional Neural Network used on Kaggle's 180 Bird Species Data Set. Within this data, there are 24,507 training images for the 180 different bird species, along with 900 validation and test images. This project was done using Google Colaboratory. Using transfer learning and API functional, I trained the CNN and ran different architectures to get the highest test accuracy possible. In the end, I was able to exceed 98% accuracy. I believe I could have surpassed this mark in my project deadline if I was able to run the full model, with more epochs, in less than an hour. On average, 5 epochs took anywhere from 2-5 hours for one model.

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

Now that I have my transfer learning model established, I'll need to make sure that the pre-established weights will be frozen so that my model doesn't need to spend any time learning the basic patterns. After some trial and error, I've determined that my model best performs when all of the weights are frozen except for the last 3 layers - which is 1 convolutional layer with a normalization and activation function. This layer will help my model learn the more complex patterns, as will the other layers that I will be adding on to my model:

```
for layer in base_model.layers[:-3]:
    layer.trainable = False
for layer in base_model.layers:
    print(layer, layer.trainable)
```

<img src = "https://user-images.githubusercontent.com/39016197/92844118-bda24000-f3a2-11ea-9cf1-cceb44fbffd2.png" width = 750 height = 550>


Finally, I'll begin building the rest of my Functional API model using the Xception model as my base. I'll also be adding 2 convolutional layers, that are followed by a dropout and average pool layer, until the model is flattened into one channel. Then I'll add a batch normalization layer, a hidden layer with 250 nodes, followed by a last dropout and our final softmax layer with 180 nodes – representing each bird species. The learning rate yielded best results when set at 0.0001 on the Adam Optimizer, the loss will be categorical crossentrophy and our standard metric will be accuracy.

```
inputs = Input(shape=(224,224,3))

base = base_model (inputs)

conv1 = Conv2D(184, (3,3), padding = 'same', activation='relu')(base)

drop1 = Dropout(0.3)(conv1)

conv2 = Conv2D(184, (3,3), padding = 'same', activation='relu')(drop1)

avgp = AveragePooling2D((3,3), padding = 'same')(conv2)

flat = Flatten()(avgp)

Bn = BatchNormalization()(flat)

hidden = Dense(250, activation='relu')(Bn)

drop2 = Dropout(0.3)(hidden)

output = Dense(180, activation='softmax')(drop2)

model = Model(inputs=inputs, outputs=output)

model.compile(tf.keras.optimizers.Adam(learning_rate=0.0001),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
```

Next, I'll print out a visual image of this architecture. While I really wanted to go for the fancy shared layers set up and split the layers a few different ways, in the end it turned out that a single path model here gave the best results.

```
plot_model(model)
```

<img src = "https://user-images.githubusercontent.com/39016197/92844585-428d5980-f3a3-11ea-926a-25b96b3c01a2.png" width = 300 height = 800>

I also ran a summary of my final model to see how the input shape is converted through the layers and how the shape of the final output looks. While there is over 25 million parameters in this model, only roughly 7.3 parameters are trainable since we froze most of the xception model's weights.

```
print('My Model with Transer Learning', model.summary())
```
<img src = "https://user-images.githubusercontent.com/39016197/92844976-ac0d6800-f3a3-11ea-8ba6-c0f37a85b861.png" width = 550 height = 600>

Finally, I'll run the model. Every time the kernal resets, the first epoch can take at least a few hours - going up to 5 hours max. While this wasn't necessarily ideal for timing, I did eventually settle for the test accuracy below (although, I guess 98% test accuracy isn't exactly settling). The training steps per epoch was set to 100 so that each image gets passed through once per epoch, as well as the 30 steps per epoch for the validation and testing. 5 epochs seemed to be the magic number, as additional epochs led to a lesser accuracy despite restoring the best weights.

```
backend.clear_session()
history = model.fit( 
    train_generator, 
    steps_per_epoch = 100,  
    epochs=5, 
    validation_data = validataion_generator, 
    validation_steps = 30,
    verbose = 1,
    callbacks=[EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights = True)])

test_loss, test_acc = model.evaluate(test_generator, steps = 30) 
               
print('test_acc:', test_acc)
```
<img src = "https://user-images.githubusercontent.com/39016197/92845529-52596d80-f3a4-11ea-9f01-697e3072b355.png" width = 850 height = 250>

```
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
epochs = range(1, len(history_dict['accuracy']) + 1)

plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, acc_values, 'bo', label = 'Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label = 'Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

<img src = "https://user-images.githubusercontent.com/39016197/92845979-ca279800-f3a4-11ea-8827-eafbea29098c.png" width = 400 height = 500>
