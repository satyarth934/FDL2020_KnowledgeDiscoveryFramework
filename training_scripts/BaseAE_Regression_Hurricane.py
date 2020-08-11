#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import model
import utils
sys.dont_write_bytecode = True
random.seed(234)

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

DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes/*/*"

NORMALIZE = True
MODEL_NAME = "baseAE_hurricane_try3_regression"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
EMBEDDING_MODEL_NAME = "baseAE_hurricane_try3"
EMBEDDING_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + EMBEDDING_MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200
BATCH_SIZE = 64
INTERPOLATE_DATA_GAP = False

cid_num_map = {"C1": 1, "C2": 2, "C3": 3, "C4": 4, "C5": 5, "TD": 6, "TS": 0}


# wind_speed in knots
def getCategory(wind_speed):
    if wind_speed <= 33:
        return 'TD'
    elif 34 <= wind_speed <= 63:
        return 'TS'
    elif 64 <= wind_speed <= 82:
        return 'C1'
    elif 83 <= wind_speed <= 95:
        return 'C2'
    elif 96 <= wind_speed <= 112:
        return 'C3'
    elif 113 <= wind_speed <= 136:
        return 'C4'
    elif wind_speed >= 137:
        return 'C5'


def classname(str):    
    file_name = str.split("/")[-1]
    wind_speed = int(file_name.split(".")[0].split("_")[-1].strip("kts"))
    return getCategory(wind_speed)


def getWindSpeed(filename):
    file_name = filename.split("/")[-1]
    wind_speed = int(file_name.split(".")[0].split("_")[-1].strip("kts"))
    return wind_speed


def customGeneratorForRegression(input_file_paths, dims, data_type):
    for i, file_path in enumerate(input_file_paths):
        wind_speed = getWindSpeed(file_path.decode("utf-8"))
        if data_type.decode("utf-8") in ["png" or "tif"]:
            img = plt.imread((file_path.decode("utf-8")))
        elif data_type.decode("utf-8") == "npy":
            img = np.load(file_path.decode("utf-8"))
        x = resize(img[:,:,:3], dims)
            
        yield x, [float(wind_speed)]
#         yield x, [tf.cast(wind_speed, tf.float32)]
        
        
def main():
    dims = (448, 448, 3)
    
    # Dataloader creation and test
    img_paths = glob.glob(DATA_PATH)
    print("len(img_paths):", len(img_paths))
    random.shuffle(img_paths)

    train_split = 0.8
    valid_split = 0.1
    test_split = 0.1
    tiny_train_subset = img_paths[:int(train_split * len(img_paths))]
    tiny_valid_subset = img_paths[int(train_split * len(img_paths)):int((train_split + valid_split) * len(img_paths))]
    test_subset = img_paths[len(img_paths) - int(test_split * len(img_paths)):]
#     tiny_train_subset, tiny_valid_subset, test_subset = splitDataset(img_paths, num_samples_per_class=150)
    
#     # More efficient data fetch pipeline
    AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
    
#     X_train = customGenerator(X_train_paths, dims)
#     X_test = customGenerator(X_test_paths, dims)
    train_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForRegression, output_types=(np.float32, np.float32), output_shapes=(dims,1), args=[tiny_train_subset, dims, "png"])
    output = list(train_dataset.take(1).as_numpy_iterator())

    print("@#@#@#@#@#@#@ 1")
    print("train_dataset:", train_dataset)
    print("len(output):", len(output))
    print("len(output[0]):", len(output[0]))
    x,y = output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max())
    print("y:", y)

    valid_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForRegression, output_types=(np.float32, np.float32), output_shapes=(dims,1), args=[tiny_valid_subset, dims, "png"])
    
    test_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForRegression, output_types=(tf.float32, tf.float32), args=[test_subset, dims, "png"])
    
#     train_dataset = tf.data.Dataset.from_tensor_slices((X_train_reshaped, X_train_reshaped))
#     test_dataset = tf.data.Dataset.from_tensor_slices((X_test_reshaped, X_test_reshaped))

    train_dataset = train_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.cache().shuffle(buffer_size=3*BATCH_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    
    valid_dataset = valid_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.cache().shuffle(buffer_size=3*BATCH_SIZE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = test_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    
#     print("train_dataset:", train_dataset)
#     print(train_dataset[0])
    output = list(train_dataset.take(1).as_numpy_iterator())

    print("@#@#@#@#@#@#@ 2")
    print("train_dataset:", train_dataset)
    print("len(output):", len(output))
    print("len(output[0]):", len(output[0]))
    x,y = output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max())
    print("y:", y.min(), y.max())
    print(x.shape, y.shape)
    print("---------------------------------")
    
    
    # ARCHITECTURE
    print("---- Reading Model ----")
    regression_model_parent = Sequential()
    regression_model = load_model(EMBEDDING_MODEL_PATH)
    print(regression_model.summary())
    
    print("model.inputs:", regression_model.inputs)
    print("conv2d_8:", regression_model.get_layer('conv2d_8').output)
    
    layer_name = 'conv2d_8'
    regression_model = Model(inputs=regression_model.inputs, 
                                 outputs=regression_model.get_layer(layer_name).output)
    
    for layer in regression_model.layers:
        layer.trainable = False
    
    regression_model_parent.add(regression_model)
    regression_model_parent.add(Flatten())
    regression_model_parent.add(Dense(256, activation='relu'))
    regression_model_parent.add(Dropout(0.1))
    regression_model_parent.add(Dense(64, activation='relu'))
    regression_model_parent.add(Dropout(0.1))
    regression_model_parent.add(Dense(1))
    print(regression_model_parent.summary())
    
    regression_model_parent.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                                        loss="mse",
                                        metrics=["mae", "mse"],)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)
    regression_model_parent.fit(train_dataset,
                                    epochs=NUM_EPOCHS,
                                    steps_per_epoch=len(tiny_train_subset) // BATCH_SIZE,
#                                     validation_split=0.2,
                                    validation_data=valid_dataset,
                                    validation_steps=len(tiny_valid_subset) // BATCH_SIZE,
                                    callbacks=[tensorboard_callback, WandbCallback()],
                                    use_multiprocessing=True)
    
    regression_model_parent.save(OUTPUT_MODEL_PATH)
    
    
if __name__ == "__main__":
    main()
