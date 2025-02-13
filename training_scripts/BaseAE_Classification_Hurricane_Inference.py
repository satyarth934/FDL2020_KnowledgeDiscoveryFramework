#!/usr/bin/env python
# coding: utf-8

from import_modules import *
# import tensorflow_addons as tfa
import sklearn.metrics

import model
import utils
import BaseAE_hurricane as aeh

sys.dont_write_bytecode = True
random.seed(234)

# import wandb
# from wandb.keras import WandbCallback
# wandb.init(config={"hyper": "parameter"})

"""
Parameters:
1. Data path
2. Normalize or not
3. Tensorboard Log Directory
4. Model path to save
"""

DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes/*/*"

NORMALIZE = True
MODEL_NAME = "baseAE_hurricane_try3_classification_80_10_10"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
# EMBEDDING_MODEL_NAME = "baseAE_hurricane_try3"
# EMBEDDING_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + EMBEDDING_MODEL_NAME
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
    test_subset = img_paths[len(img_paths) - int(test_split * len(img_paths)):][:128]
#     tiny_train_subset, tiny_valid_subset, test_subset = splitDataset(img_paths, num_samples_per_class=70)
    print("test_subset:", len(test_subset))
    test_subset = utils.getUsableImagePaths(image_paths=test_subset, data_type="png")
    print("len(usable test_subset):", len(test_subset))
    
    # More efficient data fetch pipeline
    AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
   
    test_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification, 
                                                  output_types=(tf.float32, tf.float32), 
                                                  output_shapes=(dims, len(cid_num_map)), 
                                                  args=[test_subset, dims, "png"])
    
    test_dataset = test_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    # ARCHITECTURE
    print("---- Reading Model ----")
#     classification_model_parent = Sequential()
    classification_model = load_model(OUTPUT_MODEL_PATH)
    print(classification_model.summary())
    
    print("=====================")
    print("---- PREDICTIONS ----")
    print("=====================")
    predictions = classification_model.predict(test_dataset,
                                               max_queue_size=BATCH_SIZE,
                                               callbacks=[])
    print(type(predictions))
    print("predictions.shape:", predictions.shape)
    y_preds = np.argmax(predictions, axis=1, out=None)
    print("len(y_preds):", len(y_preds))
    print("y_preds.shape:", y_preds.shape)
    
    ground_truth = np.array([tf.keras.utils.to_categorical(cid_num_map[classname(f)], num_classes=len(cid_num_map)) for f in test_subset])
#     wandb.sklearn.plot_confusion_matrix(y_true, predictions, labels=list(cid_num_map.keys()))
    print("ground_truth.shape:", ground_truth.shape)
    y_true = np.argmax(ground_truth, axis=1, out=None)
    print("len(y_true):", len(y_true))
    print("y_true.shape:", y_true.shape)
    
    con_mat_tf = tf.math.confusion_matrix(labels=y_true, predictions=y_preds).numpy()
    print("tf math\n", con_mat_tf)
    print("Accuracy tf:", np.sum(np.diag(con_mat_tf)) / np.sum(con_mat_tf))
    
    con_mat_norm = np.around(con_mat_tf.astype('float') / con_mat_tf.sum(axis=1)[:, np.newaxis], decimals=2)
    print("tf math normalized\n", con_mat_norm)
    print("Accuracy tf normed:", np.sum(np.diag(con_mat_norm)) / np.sum(con_mat_norm))
    
    w_f1score = sklearn.metrics.f1_score(y_true, y_preds, labels=None, average='weighted')
    print("Weighted F1-score:", w_f1score)

    print("=====================")
    print("---- EVALUATIONS ----")
    print("=====================")
    print("test_dataset:", test_dataset)
    test_output = list(test_dataset.take(1).as_numpy_iterator())
    print("len of test_output and test_output[0]:", len(test_output), len(test_output[0]))
    x,y = test_output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max(), "\ty:", y.min(), y.max())
    print("---------------------------------")
    eval_dict = classification_model.evaluate(test_dataset, verbose=1, return_dict=True)
    
    print("eval_dict:", eval_dict)
    
if __name__ == "__main__":
    main()
