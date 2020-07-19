# Change to current dataset
import os
import sys
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

# Imports from Colab 2
import math
import numpy as np
import pickle
import keras
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from datetime import datetime
# Import pretrained model
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Imports for Colab 6
import cv2 # Read raw image
import glob
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
from scipy import ndimage # For rotation task or
import imutils
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.preprocessing.image import Iterator


# Imports for Colorizer
from os import path
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, Reshape, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.layers import Activation, InputLayer, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import random
import tensorflow as tf

# Check to see if GPU is being used
tensorflow.test.gpu_device_name()

"""# Data Augmentation/Analysis"""

batch_size = 16

class CustomDataGenerator(data_utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=batch_size, dim=(400,400,3), shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X,y= self.__data_generation(list_IDs_temp)
        #print(X.shape,y.shape)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        for i,f in enumerate(list_IDs_temp):
            X[i,]=np.load(f)

        gray_batch=rgb2gray(X)

        X_batch=gray_batch[:,:,:]/255. #(16,400,400)
        X_batch=np.repeat(np.expand_dims(X_batch,3),3,axis=3)
        Y_batch=X[:,:,:,:]/255.
        return X_batch,Y_batch

#model.save('Models/Colorization_MODIS_Exp_Custom_' + dt_string)
