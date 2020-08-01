#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import model
import utils
sys.dont_write_bytecode = True

import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"})

"""
Parameters:
1. Data path
2. Normalize or not
3. Tensorboard Log Directory
4. Model path to save
"""
# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed_gap_filled/*"
DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes_reorganized"

NORMALIZE = True
MODEL_NAME = "baseAE_hurricane"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200
BATCH_SIZE = 64
INTERPOLATE_DATA_GAP = False


# # Check to see if GPU is being used
print(tensorflow.test.gpu_device_name())
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# # Custom data generator to read from the image paths sequence
# class CustomDataGenerator(data_utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, batch_size=32, dim=(448,448,3), shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.list_IDs = list_IDs
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X,y= self.__data_generation(list_IDs_temp)
#         #print(X.shape,y.shape)
#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim))
#         for i,f in enumerate(list_IDs_temp):
#             X[i,] = resize(plt.imread(f)[:,:,:3], self.dim)
#             while (np.sum(np.isnan(X[i])) > 0):
#                 X[i,] = resize(plt.imread(list_IDs_temp[np.random.randint(len(list_IDs_temp))])[:,:,:3], self.dim)

# #         X_batch=X
# #         Y_batch=X
#         return X, x


def main():
#     # Dataloader creation and test
#     img_paths = glob.glob(DATA_PATH)
#     print("len(img_paths):", len(img_paths))
#     random.shuffle(img_paths)

    train_test_split = 0.8
#     X_train_paths = img_paths[:int(train_test_split * len(img_paths))]
#     X_test_paths = img_paths[int(train_test_split * len(img_paths)):]

    X_train_path = DATA_PATH + "/train"
    X_test_path = DATA_PATH + "/test"
    dims = (448, 448, 3)

    # Loading Data
#     X_train = CustomDataGenerator(X_train_paths, batch_size=32, dim=dims)
#     X_test = CustomDataGenerator(X_train_paths, batch_size=32, dim=dims)

    train_generator = ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True,
                                       validation_split=(1 - train_test_split))
    
    X_train = train_generator.flow_from_directory(X_train_path, target_size=(448,448), class_mode="input", batch_size=32, seed=324, subset='training')
    X_valid = train_generator.flow_from_directory(X_train_path, target_size=(448,448), class_mode="input", batch_size=32, seed=324, subset='validation')
    X_test = ImageDataGenerator().flow_from_directory(X_test_path, target_size=(448,448), class_mode="input", batch_size=32, seed=324) 

    sample_train = next(X_train)
    print("sample train:", sample_train[0].shape, sample_train[1].shape)
    for i, img in enumerate(sample_train[0][0][:10]):
        print(np.sum(np.isnan(img)), img.min(), img.max())
    sample_test = next(X_test)
    print("sample test:", sample_test[0].shape, sample_test[1].shape)

#     # RESHAPE IMAGES TO THE DESIRED SIZE
#     X_train_reshaped = X_train[0]
#     print(X_train_reshaped[0].shape, X_train_reshaped[1].shape)
#     X_test_reshaped = X_test[0]
#     print(X_test_reshaped[0].shape, X_test_reshaped[1].shape)

#     # More efficient data fetch pipeline
#     AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
    
#     train_dataset = tf.data.Dataset.from_generator(generator=X_train, output_types=(tf.float32, tf.float32))
#     test_dataset = tf.data.Dataset.from_generator(generator=X_test, output_types=(tf.float32, tf.float32))
    
# #     train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, X_train_reshaped))
# #     test_dataset = tf.data.Dataset.from_tensor_slices((X_test_reshaped, X_test_reshaped))

#     train_dataset = train_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
#     train_dataset = train_dataset.cache().shuffle(buffer_size=3*BATCH_SIZE)
#     train_dataset = train_dataset.batch(BATCH_SIZE)
#     train_dataset = train_dataset.repeat()
#     train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

#     test_dataset = test_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
#     test_dataset = test_dataset.cache()
#     test_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    
    # ARCHITECTURE
    complete_model = model.createModel(dims)
    print(complete_model.summary())

    complete_model.compile(optimizer='rmsprop', loss='mse')

    image = np.expand_dims(X_test[0][0][0], 0)
    # image = X_test_reshaped[0:10]
    print(image.shape)

    # Define the Activation Visualization callback
    output_dir = TENSORBOARD_LOG_DIR
    callbacks = [
        ActivationsVisualizationCallback(
            validation_data=(image,),
            layers_name=['conv2d_8'],
            output_dir=output_dir,
        ),
    ]

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf_callback")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)

    complete_model.fit(X_train,
                       epochs=NUM_EPOCHS,
                       steps_per_epoch=X_train.samples // BATCH_SIZE,
#                        validation_split=0.2,
                       validation_data=X_valid,
                       validation_steps=X_valid.samples / BATCH_SIZE,
                       callbacks=[callbacks, tensorboard_callback, WandbCallback()],
                       use_multiprocessing=True
                       )

    complete_model.save(OUTPUT_MODEL_PATH)



    # ## Model Testing

    # Define the Activation Visualization explainer
    # index = np.random.randint(0, len(X_test_reshaped))
    # image = input_test[index].reshape((1, 32, 32, 3))
    # image = np.expand_dims(X_test_reshaped[index],0)
    image = X_test[0][0][:10]
#     image = X_test_reshaped[:10]
    label = image
    print('val:', image.shape)

    data = ([image])
    explainer = ExtractActivations()

    layers_of_interest = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_transpose', 'conv2d_transpose_1', 'conv2d_transpose_2', 'conv2d_transpose_3', 'conv2d_9', 'conv2d_transpose_4']
    for layer in layers_of_interest:
        grid = explainer.explain(validation_data=data, model=complete_model, layers_name=[layer])
        print(grid.shape)
        explainer.save(grid, ACTIVATION_IMG_PATH, '%s.png' % (layer))


#     # complete_model = model.load_model(OUTPUT_MODEL_PATH)
#     for i in range(10):
#         index = np.random.randint(0, len(X_test_reshaped))

#         X_test_im = np.expand_dims(X_test_reshaped[index], 0)
#         out_image = np.squeeze(complete_model.predict(X_test_im))

#         im_min = out_image.min(axis=(0, 1), keepdims=True)
#         im_max = out_image.max(axis=(0, 1), keepdims=True)
#         out_image = (out_image - im_min) / (im_max - im_min)

#         print("Orig ", np.min(X_test_im), np.max(X_test_im))
#         print("Gen ", np.min(out_image), np.max(out_image))
#         fig = plt.figure()
#         plt.subplot(1, 3, 1)
#         plt.imshow(X_test_reshaped[index])
#         plt.subplot(1, 3, 2)
#         plt.imshow(np.squeeze(X_test_im))
#         plt.subplot(1, 3, 3)
#         plt.imshow(out_image)
#         plt.show()


if __name__ == "__main__":
    main()
