#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import model



"""
Parameters:
1. Data path
2. Normalize or not
3. Tensorboard Log Directory
4. Model path to save
"""
# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*"
NORMALIZE = True
MODEL_NAME = "baseAE_mean"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200


# Check to see if GPU is being used
print(tensorflow.test.gpu_device_name())
print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


# # Dataloader creation and test
# NEW MODIS DATASET

# img_paths = glob.glob(DATA_PATH)
# img_paths = glob.glob(DATA_PATH) # <- needs to be normalized
img_paths = glob.glob(DATA_PATH)  # <- needs to be normalized
print("len(img_paths):", len(img_paths))
random.shuffle(img_paths)

train_test_split = 0.8
X_train_paths = img_paths[:int(train_test_split * len(img_paths))]
X_test_paths = img_paths[int(train_test_split * len(img_paths)):]
np.save(PATH_LIST_LOCATION, [X_train_paths, X_test_paths])

dims = (448, 448, 3)

# Loading Training Data
X_train = np.empty((len(X_train_paths), *dims))
for i, p in enumerate(X_train_paths):
    X_train[i, :, :, :] = np.load(p)

# Loading Testing Data
X_test = np.empty((len(X_test_paths), *dims))
for i, p in enumerate(X_test_paths):
    X_test[i, :, :, :] = np.load(p)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# To check what percentage of pixels are 'nan'
print(np.sum(np.isnan(X_train)) / np.prod(X_train.shape))
print(np.sum(np.isnan(X_test)) / np.prod(X_test.shape))

# Checking min max to see if normalization is needed or not
print("Before normalization")
print(np.nanmin(X_train), np.nanmax(X_train))
print(np.nanmin(X_test), np.nanmax(X_test))

# Normalize Inputs


def normalize(mat):
    valid_cells = np.invert(np.isnan(mat))
    normalized = np.subtract(mat, np.nanmin(mat), where=valid_cells) / (np.nanmax(mat) - np.nanmin(mat))
    return normalized


X_train = normalize(X_train)
X_test = normalize(X_test)

# Checking min max after normalization
print("After normalization")
print(np.nanmin(X_train), np.nanmax(X_train))
print(np.nanmin(X_test), np.nanmax(X_test))

# Set nan values to 0
# X_train[np.isnan(X_train)] = 0.0
# X_test[np.isnan(X_test)] = 0.0


X_train_reshaped = X_train
del X_train
X_test_reshaped = X_test
del X_test

batch_size = 64

AUTOTUNE = tensorflow.data.experimental.AUTOTUNE


def convert(image, label):
    image = tensorflow.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
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


# In[6]:


def batch_lab(batch_size, data_generator, data):  # Does basically nothing, but just to help with later tasks
  for batch in data_generator.flow(data, batch_size=batch_size):
    batch = resize(batch, (batch_size, *dims))
#     print(np.max(batch),np.min(batch))
    yield batch, batch


# # Model creation

# In[7]:


complete_model = model.createModel(dims)
print(complete_model.summary())

complete_model.compile(optimizer='rmsprop', loss='mse')



image = np.expand_dims(X_test_reshaped[0], 0)
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


complete_model.fit(train_dataset,
                   epochs=NUM_EPOCHS,
                   steps_per_epoch=len(X_train_reshaped) / batch_size,
                   validation_data=test_dataset,
                   validation_steps=len(X_test_reshaped) / batch_size,
                   callbacks=[callbacks, tensorboard_callback],
                   use_multiprocessing=True
                   )

complete_model.save(OUTPUT_MODEL_PATH)



# ## Model Testing

# Define the Activation Visualization explainer
# index = np.random.randint(0, len(X_test_reshaped))
# image = input_test[index].reshape((1, 32, 32, 3))
# image = np.expand_dims(X_test_reshaped[index],0)
image = X_test_reshaped[:10]
label = image
print('val:', image.shape)

data = ([image])
explainer = ExtractActivations()

layers_of_interest = ['conv2d_1']
grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_1'])
print(grid.shape)
explainer.save(grid, ACTIVATION_IMG_PATH, 'conv2d_1.png')

grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_2'])
print(grid.shape)
explainer.save(grid, ACTIVATION_IMG_PATH, 'conv2d_2.png')

grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_3'])
print(grid.shape)
explainer.save(grid, ACTIVATION_IMG_PATH, 'conv2d_3.png')

grid = explainer.explain(validation_data=data, model=complete_model, layers_name=['conv2d_8'])
print(grid.shape)
explainer.save(grid, ACTIVATION_IMG_PATH, 'conv2d_8.png')


# In[ ]:


for i in range(10):
    index = np.random.randint(0, len(X_test_reshaped))

    X_test_im = np.expand_dims(X_test_reshaped[index], 0)
    out_image = np.squeeze(complete_model.predict(X_test_im))

    im_min = out_image.min(axis=(0, 1), keepdims=True)
    im_max = out_image.max(axis=(0, 1), keepdims=True)
    out_image = (out_image - im_min) / (im_max - im_min)

    print("Orig ", np.min(X_test_im), np.max(X_test_im))
    print("Gen ", np.min(out_image), np.max(out_image))
    fig = plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(X_test_reshaped[index])
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(X_test_im))
    plt.subplot(1, 3, 3)
    plt.imshow(out_image)
    plt.show()




