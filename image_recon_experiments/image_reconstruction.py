# -*- coding: utf-8 -*-

# Change to current dataset
import sys
sys.dont_write_bytecode = True

# Imports from Colab 2
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, InputLayer, BatchNormalization
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, Reshape, Conv2DTranspose, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.preprocessing.image import Iterator
# Import pretrained model
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Imports for Colab 6
import cv2  # Read raw image
import glob
from datetime import datetime
# from google.colab.patches import cv2_imshow
from matplotlib import pyplot as plt
from scipy import ndimage  # For rotation task or
import imutils


# Imports for Colorizer
from os import path
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.io import imsave
import random

# Imports for Autoencoder
import utils
import network_definition as nd
from utils_data import parser, from_tfrecords
from keras.layers import Input
from keras.models import Model
# Check to see if GPU is being used
tf.test.gpu_device_name()

dims = (400, 400, 3)
bool_generate = False
BATCH_SIZE = 8

root_dir = "/home/satyarth934/Projects/NASA_FDL_2020/Datasets/MODIS_MCD43A4/"
train_tfrecords_dir = root_dir + "tfrecords/train/train.tfrecords-*"
valid_tfrecords_dir = root_dir + "tfrecords/valid/valid.tfrecords-*"
test_tfrecords_dir = root_dir + "tfrecords/test/test.tfrecords-*"

train_tfrecords = glob.glob(train_tfrecords_dir)
valid_tfrecords = glob.glob(valid_tfrecords_dir)
test_tfrecords = glob.glob(test_tfrecords_dir)

"""# Dataloader creation and test"""

train_dataset, train_gt = from_tfrecords(records_globs=train_tfrecords,
                               split="train",
                               batch_size=BATCH_SIZE)

print(train_dataset.shape)
print(train_gt.shape)
sys.exit(0)

valid_dataset = from_tfrecords(records_globs=valid_tfrecords,
                               split="valid",
                               batch_size=BATCH_SIZE)

print("LEN TRAINING:", len(train_tfrecords))
print("LEN VALID:", len(valid_tfrecords))

"""# Model"""

#model = nd.Autoencoder(is_fusion=False,depth_after_fusion=256)
# print("---------------------------------------------")
# print("Encoder:")
# print(model.encoder.summary())
#print("Encoder out shape:",model.encoder.output_shape)
# print("Decoder:")
# print(model.decoder.summary())
#print("Decoder out shape:",model.decoder.output_shape)

# image=Input(shape=dims)
# encoder_model = encoder(dims)
# print(encoder_model.output_shape)
# decoder_model = decoder(encoder_model.output_shape[-3:])
# print(decoder_model.output_shape)

complete_model = nd.network(input_shape=dims)
# complete_model = decoder_model(encoder_model.output)
print(complete_model.summary())


#model = model.build()
# complete_model=tf.keras.Model(inputs=model.encoder.input,outputs=
# print(model.summary())
"""# Model Training"""

# model.compile(optimizer = 'rmsprop', loss = 'mse', metrics = ['accuracy'])
complete_model.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])


callback_earlystop = EarlyStopping(monitor='loss', patience=5)

checkpoint_filepath = root_dir + 'Models/checkpoint/vanilla_ae_{epoch:04d}.h5'
callback_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    period=1)


# model.summary()
complete_model.fit(train_dataset,
                  epochs=100,
                  steps_per_epoch=len(train_tfrecords) / BATCH_SIZE,
                  callbacks=[callback_earlystop, callback_checkpoint])
sys.exit(0)

now = datetime.now()
dt_string = now.strftime("%d_%m_%H_%M")
print(dt_string)
complete_model.save('Models/vanilla_ae_MODIS_Exp_Custom_' + dt_string)
