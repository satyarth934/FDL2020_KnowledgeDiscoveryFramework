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
MODEL_NAME = "baseAE_hurricane_classification_noselfsup"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
EMBEDDING_MODEL_NAME = "baseAE_hurricane_try2"
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


def customGeneratorForClassification(input_file_paths, dims, data_type):
    for i, file_path in enumerate(input_file_paths):
        cid = classname(file_path.decode("utf-8"))
        if data_type.decode("utf-8") in ["png" or "tif"]:
            img = plt.imread((file_path.decode("utf-8")))
        elif data_type.decode("utf-8") == "npy":
            img = np.load(file_path.decode("utf-8"))
        x = resize(img[:,:,:3], dims)
            
        yield x, tf.keras.utils.to_categorical(cid_num_map[cid], num_classes=len(cid_num_map))


def splitDataset(img_paths, num_samples_per_class=70):
    tiny_train_subset = []
    class_count = {}

    for imgpath in tqdm(img_paths):
        cid = classname(imgpath)
        if cid not in class_count:
            class_count[cid] = 1
            tiny_train_subset.append(imgpath)
        elif class_count[cid] >= num_samples_per_class:
            continue
        else:
            class_count[cid] += 1
            tiny_train_subset.append(imgpath)

    pprint(class_count)

    test_subset = list(set(img_paths) - set(tiny_train_subset))

    tiny_valid_subset = random.sample(tiny_train_subset, 
                                      int(0.2 * len(tiny_train_subset)))
    tiny_train_subset = list(set(tiny_train_subset) - set(tiny_valid_subset))

    print("len(tiny_train_subset):", len(tiny_train_subset))
    print("len(tiny_valid_subset):", len(tiny_valid_subset))
    print("len(test_subset):", len(test_subset))
    
    return (tiny_train_subset, tiny_valid_subset, test_subset)
        
        
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
    train_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification, output_types=(np.float32, np.float32), output_shapes=(dims, len(cid_num_map)), args=[tiny_train_subset, dims, "png"])
    output = list(train_dataset.take(1).as_numpy_iterator())

    print("@#@#@#@#@#@#@ 1")
    print("train_dataset:", train_dataset)
    print("len(output):", len(output))
    print("len(output[0]):", len(output[0]))
    x,y = output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max())
    print("y:", y)

    valid_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification, output_types=(np.float32, np.float32), output_shapes=(dims, len(cid_num_map)), args=[tiny_valid_subset, dims, "png"])
    
    test_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification, output_types=(tf.float32, tf.float32), args=[test_subset, dims, "png"])
    
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
    classification_model_parent = Sequential()
#     classification_model = load_model(EMBEDDING_MODEL_PATH)
    classification_model = model.createModel(dims)
    print(classification_model.summary())
    
    print("model.inputs:", classification_model.inputs)
    print("conv2d_8:", classification_model.get_layer('conv2d_8').output)
    
    layer_name = 'conv2d_8'
    classification_model = Model(inputs=classification_model.inputs,
                                 outputs=classification_model.get_layer(layer_name).output)
    
#     for layer in classification_model.layers:
#         layer.trainable = True
    
    classification_model_parent.add(classification_model)
    classification_model_parent.add(Flatten())
    classification_model_parent.add(Dense(256, activation='relu'))
    classification_model_parent.add(Dropout(0.1))
    classification_model_parent.add(Dense(64, activation='relu'))
    classification_model_parent.add(Dropout(0.1))
    classification_model_parent.add(Dense(len(cid_num_map), activation="softmax"))
    print(classification_model_parent.summary())
    
    classification_model_parent.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                                        loss="categorical_crossentropy",
                                        metrics=["accuracy"],)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)
    classification_model_parent.fit(train_dataset,
                                    epochs=NUM_EPOCHS,
                                    steps_per_epoch=len(tiny_train_subset) // BATCH_SIZE,
#                                     validation_split=0.2,
                                    validation_data=valid_dataset,
                                    validation_steps=len(tiny_valid_subset) // BATCH_SIZE,
                                    callbacks=[tensorboard_callback, WandbCallback()],
                                    use_multiprocessing=True)
    
    classification_model_parent.save(OUTPUT_MODEL_PATH)
    
    
if __name__ == "__main__":
    main()
