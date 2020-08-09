#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import tensorflow_addons as tfa

import sklearn
import model
import utils
import BaseAE_hurricane as aeh
sys.dont_write_bytecode = True
random.seed(234)

import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"})

############
# PARAMETERS
############
DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed_gap_filled_again/*"

NORMALIZE = True
MODEL_NAME = "baseAE_median_localRandom_in_swatch_classification"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
EMBEDDING_MODEL_NAME = "baseAE_median_localRandom_in_swatch"
EMBEDDING_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + EMBEDDING_MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 2
BATCH_SIZE = 32
INTERPOLATE_DATA_GAP = False
CUSTOM_OBJECTS = False

MODIS_DUST_GT_PATH = "/home/satyarth934/data/modis_data_products/MODIS_Dust_Events_2010_2020_h16v7.pkl"
DUST_GT = pickle.load(open(MODIS_DUST_GT_PATH, 'rb'))

DUST_GT_2 = {}
for k in DUST_GT:
    DUST_GT_2[k] = 0 if DUST_GT[k]==0 else 1

DUST_GT_3LABELS = {0:"Dust", 1:"Hazy", 2:"No_Dust"}
DUST_GT_2LABELS = {0:"Dust", 1:"No_Dust"}
# DUST_GT_CID_NUM_MAP = {"Dust": 0, "Hazy": 1, "NoDust": 2}
INPUT_DATA_TYPE = "npy"




def classname(str):
    doy = str.split('/')[-1].split('.')[1][1:]
    
    return DUST_GT[doy] if DUST_GT[doy]==0 else 1


def customGeneratorForClassification(input_file_paths, dims, data_type):
    for i, file_path in enumerate(input_file_paths):
        cid = classname(file_path.decode("utf-8"))
        if data_type.decode("utf-8") in ["png" or "tif"]:
            img = plt.imread((file_path.decode("utf-8")))
        elif data_type.decode("utf-8") == "npy":
            img = np.load(file_path.decode("utf-8"))
        x = resize(img[:,:,:3], dims)
            
        yield x, tf.keras.utils.to_categorical(cid, num_classes=len(DUST_GT_2LABELS))


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
    img_paths = glob.glob(DATA_PATH)[:1024]
    print("len(img_paths):", len(img_paths))
    img_paths = utils.getUsableImagePaths(image_paths=img_paths, data_type="npy")
    print("len(usable img_paths):", len(img_paths))
    
    # Select only the filenames that are labelled.
    img_paths = [f for f in img_paths if int(f.split("/")[-1].split(".")[1][1:]) < 2015248]
    
    random.shuffle(img_paths)

    train_split = 0.8
    valid_split = 0.1
    test_split = 0.1
    tiny_train_subset = img_paths[:int(train_split * len(img_paths))]
    tiny_valid_subset = img_paths[int(train_split * len(img_paths)):int((train_split + valid_split) * len(img_paths))]
    test_subset = img_paths[len(img_paths) - int(test_split * len(img_paths)):]
#     tiny_train_subset, tiny_valid_subset, test_subset = splitDataset(img_paths, num_samples_per_class=150)
    
    
    # More efficient data fetch pipeline
    AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
    
    train_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification,
                                                   output_types=(np.float32, np.float32), 
                                                   output_shapes=(dims, len(DUST_GT_2LABELS)), 
                                                   args=[tiny_train_subset, dims, INPUT_DATA_TYPE])
    output = list(train_dataset.take(1).as_numpy_iterator())

    print("@#@#@#@#@#@#@ 1")
    print("train_dataset:", train_dataset)
    print("len of output and output[0]:", len(output), len(output[0]))
    x,y = output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max(), "\ty:", y)

    valid_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification,
                                                   output_types=(np.float32, np.float32), 
                                                   output_shapes=(dims, len(DUST_GT_2LABELS)), 
                                                   args=[tiny_valid_subset, dims, INPUT_DATA_TYPE])
    
    test_dataset = tf.data.Dataset.from_generator(generator=customGeneratorForClassification, 
                                                  output_types=(tf.float32, tf.float32), 
                                                  output_shapes=(dims, len(DUST_GT_2LABELS)), 
                                                  args=[test_subset, dims, INPUT_DATA_TYPE])
    
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
    test_dataset = test_dataset.cache().batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    train_output = list(train_dataset.take(1).as_numpy_iterator())
    valid_output = list(valid_dataset.take(1).as_numpy_iterator())
    test_output = list(test_dataset.take(1).as_numpy_iterator())

    print("@#@#@#@#@#@#@ 2")
    print("train_dataset:", train_dataset)
    print("len of train_output and train_output[0]:", len(train_output), len(train_output[0]))
    x,y = train_output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max(), "\ty:", y.min(), y.max())
    print("---------------------------------")
    print("valid_dataset:", valid_dataset)
    print("len of valid_output and valid_output[0]:", len(valid_output), len(valid_output[0]))
    x,y = valid_output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max(), "\ty:", y.min(), y.max())
    print("---------------------------------")
    print("test_dataset:", test_dataset)
    print("len of test_output and test_output[0]:", len(test_output), len(test_output[0]))
    x,y = test_output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max(), "\ty:", y.min(), y.max())
    print("---------------------------------")
    
    
    # ARCHITECTURE
    print("---- Reading Model ----")
    classification_model = Sequential()
    if CUSTOM_OBJECTS:
        embedding_model = load_model(EMBEDDING_MODEL_PATH, 
                                     custom_objects={'ssim_loss':aeh.ssim_loss})
    else:
        embedding_model = load_model(EMBEDDING_MODEL_PATH)
        
    print(embedding_model.summary())
    
    print("model.inputs:", embedding_model.inputs)
    print("conv2d_8:", embedding_model.get_layer('conv2d_8').output)
    
    layer_name = 'conv2d_8'
    embedding_model = Model(inputs=embedding_model.inputs, 
                            outputs=embedding_model.get_layer(layer_name).output)
    
    for layer in embedding_model.layers:
        layer.trainable = False
    
    classification_model.add(embedding_model)
    classification_model.add(Flatten())
    classification_model.add(Dense(256, activation='relu'))
    classification_model.add(Dropout(0.1))
    classification_model.add(Dense(64, activation='relu'))
    classification_model.add(Dropout(0.1))
    classification_model.add(Dense(len(DUST_GT_2LABELS), activation="softmax"))
    print(classification_model.summary())
    
    METRICS = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'), 
        tf.keras.metrics.BinaryAccuracy(name='accuracy_binary'),
        tf.keras.metrics.CategoricalAccuracy(name='accuracy_categorical'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
#         tfa.metrics.F1Score(num_classes=len(DUST_GT_2LABELS), average='weighted', name='weighted_f1_score'),
    ]
    classification_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                                 loss="categorical_crossentropy",
#                                  metrics=["categorical_accuracy"])
                                 metrics=METRICS)
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)
    classification_model.fit(train_dataset,
                             epochs=NUM_EPOCHS,
                             steps_per_epoch=len(tiny_train_subset) // BATCH_SIZE,
#                              validation_split=0.2,
                             validation_data=valid_dataset,
                             validation_steps=len(tiny_valid_subset) // BATCH_SIZE,
                             callbacks=[tensorboard_callback, WandbCallback()],
                             use_multiprocessing=True)
    
    print("Saving the trained model...")
    classification_model.save(OUTPUT_MODEL_PATH)
    print("==----------==----------==----------==----------==")
    
#     print("========= READING THE MODEL =========")
#     classification_model = load_model("delete")
#     print(classification_model.summary())
    
    print("========= PREDICTION =========")
    predictions = classification_model.predict(test_dataset)
    print("predictions.shape:", predictions.shape)
    y_preds = np.argmax(predictions, axis=1, out=None)
    print("len and shape of y_preds:", len(y_preds), y_preds.shape)
    
    ground_truth = np.array([tf.keras.utils.to_categorical(classname(f), num_classes=len(DUST_GT_2LABELS))
                             for f in test_subset])
    print("ground_truth.shape:", ground_truth.shape)
    y_true = np.argmax(ground_truth, axis=1, out=None)
    print("len and shape of y_true:", len(y_true), y_true.shape)
    
    con_mat_tf = tf.math.confusion_matrix(labels=y_true, predictions=y_preds).numpy()
    print("tf math\n", con_mat_tf)
    print("Accuracy tf:", np.sum(np.diag(con_mat_tf)) / np.sum(con_mat_tf))
    
    w_f1score = sklearn.metrics.f1_score(y_true, y_preds, labels=None, average='weighted')
    print("Weighted F1-score:", w_f1score)
    
    
    print("========= EVALUATION =========")
    eval_dict = classification_model.evaluate(test_dataset, 
                                              batch_size=BATCH_SIZE, 
                                              verbose=1, return_dict=True)
    print("eval_dict:", eval_dict)
    
    
    
    
if __name__ == "__main__":
    main()
