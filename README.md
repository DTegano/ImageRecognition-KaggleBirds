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
# Importing the Data


