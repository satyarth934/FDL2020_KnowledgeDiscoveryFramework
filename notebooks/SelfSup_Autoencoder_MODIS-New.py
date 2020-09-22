#!/usr/bin/env python
# coding: utf-8
 
# Summary of this notebook: ...
# 
# Definition of Done: ...

# # Imports
# 


# Imports from Colab 2
import math
import numpy as np
import pickle
# import keras
import tensorflow
from pprint import pprint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Import model architecture
from tensorflow.keras.applications import VGG16


# In[3]:


# Imports for Colab 6
import os
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import cv2 # Read raw image
import glob
import random
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
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, MaxPooling2D, Reshape, Conv2DTranspose, ZeroPadding2D, Add
from tensorflow.keras.layers import Activation, InputLayer, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imsave
import random
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.layers import PReLU


# In[4]:


# Check to see if GPU is being used
print(tensorflow.test.gpu_device_name())
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# # Dataloader creation and test

# In[5]:


# NEW MODIS DATASET

#input_path = sys.argv[1]
input_path = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"

img_paths = glob.glob(input_path)
print("len(img_paths):", len(img_paths))
random.shuffle(img_paths)

train_test_split = 0.8
X_train_paths = img_paths[:int(train_test_split*len(img_paths))]
X_test_paths = img_paths[int(train_test_split*len(img_paths)):]

dims=(448,448,3)

# Loading Training Data
X_train = np.empty((len(X_train_paths),*dims))
for i, p in enumerate(X_train_paths):
    X_train[i,:,:,:] = np.load(p)

# Loading Testing Data
X_test = np.empty((len(X_test_paths),*dims))
for i, p in enumerate(X_test_paths):
    X_test[i,:,:,:] = np.load(p)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# Set nan values to 0
X_train[np.isnan(X_train)] = 0.0
X_test[np.isnan(X_test)] = 0.0


# In[6]:


print(X_train.shape,X_test.shape)


# In[7]:


X_train_reshaped = X_train
del X_train
X_test_reshaped = X_test
del X_test

batch_size = 64

AUTOTUNE=tensorflow.data.experimental.AUTOTUNE

def convert(image, label):
    image = tensorflow.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
    label = tensorflow.image.convert_image_dtype(label, tf.float32)
    return image, label

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, X_train_reshaped))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test_reshaped, X_test_reshaped))

train_dataset = train_dataset.map(convert, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = test_dataset.map(convert, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.cache()
test_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)


# In[8]:


def batch_lab(batch_size,data_generator,data): # Does basically nothing, but just to help with later tasks
  for batch in data_generator.flow(data,batch_size=batch_size):
    batch=resize(batch,(batch_size,*dims))
#     print(np.max(batch),np.min(batch))
    yield batch,batch


# # Model creation

# In[9]:


complete_model=Sequential(name="complete_model")
complete_model.add(Input(shape=dims))
complete_model.add(Conv2D(32, (3, 3), padding="same", strides=2))
complete_model.add(PReLU())
complete_model.add(Conv2D(64, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2D(64, (3, 3), padding="same", strides=2))
complete_model.add(PReLU())
complete_model.add(Conv2D(128, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2D(128, (3, 3), padding="same", strides=2))
complete_model.add(PReLU())
complete_model.add(Conv2D(256, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2D(256, (3, 3), padding="same", strides=2))
complete_model.add(PReLU())
complete_model.add(Conv2D(15, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2D(5, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2DTranspose(128, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(UpSampling2D((2, 2)))
complete_model.add(Conv2DTranspose(64, (3, 3),padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2DTranspose(64, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(UpSampling2D((2, 2)))
complete_model.add(Conv2DTranspose(32, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(Conv2D(3, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(UpSampling2D((2, 2)))
complete_model.add(Conv2DTranspose(3, (3, 3), padding="same"))
complete_model.add(PReLU())
complete_model.add(UpSampling2D((2, 2)))
complete_model.add(Conv2DTranspose(3, (3, 3), activation="tanh", padding="same"))

print(complete_model.summary())


# In[ ]:


# def encoder(input_shape):

#     model = Sequential(name="encoder")
#     model.add(Input(shape=input_shape))
#     model.add(Conv2D(32, (3, 3), padding="same", strides=2))
#     model.add(PReLU())
#     model.add(Conv2D(64, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(Conv2D(64, (3, 3), padding="same", strides=2))
#     model.add(PReLU())
#     model.add(Conv2D(128, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(Conv2D(128, (3, 3), padding="same", strides=2))
#     model.add(PReLU())
#     model.add(Conv2D(256, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(Conv2D(256, (3, 3), padding="same", strides=2))
#     model.add(PReLU())
#     model.add(Conv2D(15, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(Conv2D(5, (3, 3), padding="same"))
#     model.add(PReLU())
#     return model

# def decoder(input_shape):
#     model = Sequential(name="decoder")
#     model.add(Input(shape=input_shape))
#     model.add(Conv2DTranspose(128, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2DTranspose(64, (3, 3),padding="same"))
#     model.add(PReLU())
#     model.add(Conv2DTranspose(64, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2DTranspose(32, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(Conv2D(3, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2DTranspose(3, (3, 3), padding="same"))
#     model.add(PReLU())
#     model.add(UpSampling2D((2, 2)))
#     model.add(Conv2DTranspose(3, (3, 3), activation="tanh", padding="same"))
#     return model

# encoder_model=encoder(dims)
# decoder_model=decoder(encoder_model.output_shape[1:])

# complete_model=Sequential(name="complete_model")
# # complete_model.add(Input(shape=dims))
# complete_model.add(encoder_model)
# complete_model.add(decoder_model)

# complete_model.build(input_shape=(None,*dims))
# print(complete_model.summary())


# In[ ]:


# viz_model = Sequential()
# for i in complete_model.submodules:
#     viz_model.add(i)

# pprint(viz_model.layers)


# # Model Training

# In[10]:


# complete_model.compile(optimizer='rmsprop', loss='mse')
complete_model.compile(optimizer='rmsprop', loss='mse')
# complete_model.summary()


# from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback
# # Define the Activation Visualization callback
# output_dir = './visualizations'
# callbacks = [
#     ActivationsVisualizationCallback(
#         validation_data=(X_test, X_test),
#         layers_name=['conv2d_transpose_2'],
#         output_dir=output_dir,
#     ),
# ]

# In[11]:


from tf_explain.callbacks.activations_visualization import ActivationsVisualizationCallback

image = np.expand_dims(X_test_reshaped[0],0)
# image = X_test_reshaped[0:10]
print(image.shape)
plt.figure(0)
plt.imshow(image[0])
plt.show()

# Define the Activation Visualization callback
# output_dir = './visualizations_modis'
output_dir = './modis_logs'
callbacks = [
    ActivationsVisualizationCallback(
        validation_data=(image,),
        layers_name=['conv2d_8'],
        output_dir=output_dir,
    ),
]

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf_callback")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./modis_logs")


# In[12]:


complete_model.fit(train_dataset,
                    epochs=5,
                    steps_per_epoch=len(X_train_reshaped)/batch_size,
                    validation_data=test_dataset,
                    validation_steps=len(X_test_reshaped)/batch_size,
                    callbacks=[callbacks, tensorboard_callback],
                    use_multiprocessing=True
                  )


# In[ ]:


# complete_model.save('../ssd/proxy_models/ae_epoch100_ucmerced')


# ## Model Testing

# In[ ]:


from tf_explain.core.activations import ExtractActivations

# Define the Activation Visualization explainer
index = np.random.randint(0,len(X_test_reshaped))
# image = input_test[index].reshape((1, 32, 32, 3))
# image = np.expand_dims(X_test_reshaped[index],0)
image = X_test_reshaped[index:index+10]
label = image
print('val:', image.shape)

data = ([image])
explainer = ExtractActivations()

layers_of_interest = ['conv2d_1']
grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_1'])
print(grid.shape)
explainer.save(grid, '.', 'conv2d_1.png')

grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_2'])
print(grid.shape)
explainer.save(grid, '.', 'conv2d_2.png')

grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_3'])
print(grid.shape)
explainer.save(grid, '.', 'conv2d_3.png')

grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_8'])
print(grid.shape)
explainer.save(grid, '.', 'conv2d_8.png')


# In[ ]:


for i in range(10):
    index=np.random.randint(0,len(X_test_reshaped))

    X_test_im=np.expand_dims(X_test_reshaped[index],0)
    out_image=np.squeeze(complete_model.predict(X_test_im))
    
    im_min=out_image.min(axis=(0, 1), keepdims=True)
    im_max=out_image.max(axis=(0, 1), keepdims=True)
    out_image=(out_image-im_min)/(im_max-im_min)
    
    
    print("Orig ",np.min(X_test_im),np.max(X_test_im))
    print("Gen ",np.min(out_image),np.max(out_image))
    fig=plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(X_test_reshaped[index])
    plt.subplot(1,3,2)
    plt.imshow(np.squeeze(X_test_im))
    plt.subplot(1,3,3)
    plt.imshow(out_image)
    plt.show()


# ## Modifying Loss Function

# In[ ]:


model.save('../ssd/proxy_models/ae_epoch100_customloss_ucmerced')

