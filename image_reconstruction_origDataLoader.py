# Change to current dataset
import os
import sys
import glob
from datetime import datetime


# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import tensorflow as tf
from tensorflow.keras.applications import MobileNet, ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, Conv2DTranspose, UpSampling2D, AveragePooling2D, Reshape, Conv2DTranspose, ZeroPadding2D, Add, Activation, InputLayer, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import data_utils
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import utils
import network_definition as nd

# Check to see if GPU is being used
tf.test.gpu_device_name()


#dataset = "Datasets/MODIS_MCD43A4/Globe/training_set/"
dataset = "/home/satyarth934/Projects/NASA_FDL_2020/Datasets/MODIS_MCD43A4/Globe/training_set/"
# dataset = "MODIS_MCD43A4/Globe/training_set/2020051"
dataPath = dataset
modelName = "KDF_modis"

print(os.path.abspath(os.curdir))
print(os.listdir(dataPath))

image_globs = glob.glob(dataPath + '*/np_arrays/*.npy')
print(len(image_globs))
print("-----------------------------------------------------------------------------")

val_ratio = int(0.1 * len(image_globs))

val_image_globs = image_globs[:val_ratio]
train_image_globs = image_globs[val_ratio:]

# val_image_globs=glob.glob(dataPath+'/*.npy')

print(len(train_image_globs))
print(train_image_globs[0])

"""# Dataloader creation and test"""

# dims=(500,500,3)
dims = (400, 400, 3)
bool_generate = False
# X_train=np.empty((0,*dims))
# chunk_prefix='Datasets/MODIS_MCD43A4/Globe/training_set/chunks/'


# train_dataGenerator = ImageDataGenerator(rotation_range=20,
#                                    width_shift_range=0.1,
#                                    height_shift_range=0.1,
#                                    zoom_range=0.1,
#                                    horizontal_flip=True)

batch_size = 32
print("LEN TRAINING:", len(train_image_globs))
train_dataGenerator = utils.CustomDataGenerator(train_image_globs, batch_size=batch_size)

# def image_a_b_gen(batch_size):
#     for batch in train_dataGenerator.flow(dataset,
#                                           batch_size=batch_size,
#                                           target_size=(400, 400),
#                                           color_mode="rgb",
#                                           class_mode="input",
#                                           shuffle=True,
#                                           seed=42
#                                           ):
#         # print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>", len(batch), batch[0].shape, batch[1].shape)
#         gray_batch = rgb2gray(batch[0])
#
#         X_batch = gray_batch[:,:,:]/255. # Gray
#         Y_batch = batch[0][:,:,:,:]/255. # RGB
#         print(np.max(X_batch),np.min(X_batch),np.max(Y_batch),np.min(Y_batch))
#         yield (X_batch.reshape(X_batch.shape+(1,)), Y_batch)

"""# Model creation"""

"""
model = Sequential()
# model.add(InputLayer(input_shape=(256, 256, 1)))
model.add(InputLayer(input_shape=(400, 400, 3)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(AveragePooling2D(pool_size=(2,2),padding='valid'))
model.add(AveragePooling2D(pool_size=(3,3),padding='valid'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
# model.add(AveragePooling2D(pool_size=(2,2),padding='valid'))
model.add(AveragePooling2D(pool_size=(3,3),padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(AveragePooling2D(pool_size=(3,3),padding='valid'))
# model.add(AveragePooling2D(pool_size=(2,2),padding='valid'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2))
# model.add(AveragePooling2D(pool_size=(3,3),padding='valid'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# model.add(AveragePooling2D(pool_size=(3,3),padding='valid'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2))) # extra
# model.add(UpSampling2D((3, 3))) # extra
model.add(ZeroPadding2D((1, 1))) # extra
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
model.add(UpSampling2D((2, 2))) # extra
model.add(UpSampling2D((2, 2))) # extra
model.add(Conv2D(3, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
"""

dims = (400, 400, 3)
complete_model = nd.network(input_shape=dims)
print(complete_model.summary())

"""# Model Training"""

# model.compile(optimizer='rmsprop', loss='mse', metrics = ['accuracy'])
complete_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])



callback_earlystop = EarlyStopping(monitor='loss', patience=5)

checkpoint_filepath = 'Models/checkpoint/model2_{epoch:04d}.h5'
callback_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    period=5)


complete_model.fit_generator(train_dataGenerator,
                    epochs=100,
                    steps_per_epoch=len(train_image_globs) / batch_size,
                    callbacks=[callback_earlystop, callback_checkpoint],
                    use_multiprocessing=True,
                    workers=10,
                    max_queue_size=10)

now = datetime.now()
dt_string = now.strftime("%d_%m_%H_%M")
print(dt_string)
model.save('Models/AE_MODIS_Exp_Custom_' + dt_string)
