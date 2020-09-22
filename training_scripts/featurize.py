from import_modules import *
sys.dont_write_bytecode = True
random.seed(a=13521)

import utils
import BaseAE_UC_Merced as aeucm
import BaseAE_hurricane as aeh


# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*" # <- needs to be normalized
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*" # <- needs to be normalized
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed_gap_filled/*"
DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes_2019/*/*"
# DATA_PATH = "/home/satyarth934/data/proxy_data/UCMerced_LandUse/Images/*/*"
NORMALIZE = True

MODEL_NAME = "baseAE_hurricane_try3"
BASE_DIR = "/home/satyarth934/code/FDL_2020/"

OUTPUT_MODEL_PATH = BASE_DIR + "Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = BASE_DIR + "tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = BASE_DIR + "activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = BASE_DIR + "activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

FEATURE_DIR = BASE_DIR + "Features/" + MODEL_NAME
subprocess.call("mkdir -p " + FEATURE_DIR, shell=True)
PATH_LIST = FEATURE_DIR + "/filenames.pkl"
FEATURES_OUTPUT = FEATURE_DIR + "/features.pkl"

NUM_EPOCHS = 200
INTERPOLATE_DATA_GAP = False
IMAGE_TYPE = "png"

cust_obj_dict = {"baseAE_hurricane_try3_ssim": ('ssim_loss', aeh.ssim_loss), 
                 "baseAE_hurricane_try3_ssimms": ('ssim_loss_ms', aeh.ssim_loss_ms)}

# # Function to featurize the input
# def extract_features(img_array, model, layer_names):
#     outputs = [layer.output for layer in model.layers if layer.name in layer_names]          # all layer outputs
# #     print(len(outputs))
    
#     functor = K.function([model.input], outputs)   # evaluation function
# #     print(functor)
    
#     # Testing
#     layer_outs = functor([img_array])
# #     print(len(layer_outs), layer_outs[0].shape)
#     feature_list = layer_outs[0]
    
#     flat_features = [f.flatten() for f in feature_list]
#     normed_features = [f/np.linalg.norm(f) for f in flat_features]
    
#     return normed_features


def featurize():
    print("---- Reading Data ----")
    img_paths = glob.glob(DATA_PATH)

    print("len(img_paths):", len(img_paths))
    random.shuffle(img_paths)

#     train_test_split = 0.8
#     X_test_paths = img_paths[int(train_test_split * len(img_paths)):]
    train_split = 0.6
    valid_split = 0.2
    test_split = 0.2
    X_train_paths = img_paths[:int(train_split * len(img_paths))]
    X_valid_paths = img_paths[int(train_split * len(img_paths)):int((train_split + valid_split) * len(img_paths))]
    X_test_paths = img_paths[len(img_paths) - int(test_split * len(img_paths)):]

    dims = (448, 448, 3)

    print("---- Reading Model ----")
    cust_obj_label = None if MODEL_NAME not in cust_obj_dict else cust_obj_dict[MODEL_NAME][0]
    cust_obj_fn = None if MODEL_NAME not in cust_obj_dict else cust_obj_dict[MODEL_NAME][1]
    model = load_model(OUTPUT_MODEL_PATH, custom_objects={cust_obj_label:cust_obj_fn})
#     model = load_model(OUTPUT_MODEL_PATH)
    print(model.summary())
    
    print("---- Featurizing Data ----")
#     feature_list = extract_features(img_array=X_test, model=model, layer_names=['conv2d_8'])
#     print("feature_list shape:", len(feature_list), feature_list[0].shape)
    AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
    test_dataset = tf.data.Dataset.from_generator(generator=aeh.customGenerator, output_types=(tf.float32, tf.float32), args=[X_test_paths, dims, IMAGE_TYPE])
    
    test_dataset = test_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache().batch(64)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    
    print("@#@#@#@#@#@#@ 1")
    output = list(test_dataset.take(1).as_numpy_iterator())
    print("train_dataset:", test_dataset)
    print("len(output):", len(output))
    print("len(output[0]):", len(output[0]))
    x,y = output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max())
    print("y:", y.min(), y.max())
    print(x.shape, y.shape)
    print("---------------------------------")
    
    print("model.inputs:", model.inputs)
    print("conv2d_8:", model.get_layer('conv2d_8').output)
    
    layer_name = 'conv2d_8'
    intermediate_layer_model = Model(inputs=model.inputs, 
                      outputs=model.get_layer(layer_name).output)
    features = intermediate_layer_model.predict(test_dataset, 
                                 max_queue_size=64)
    print("features.shape:", features.shape)
    flat_features = [f.flatten() for f in features]
    print("flat_features.shape:", len(flat_features), flat_features[0].shape)
    feature_list = [f/np.linalg.norm(f) for f in flat_features]
    print("feature_list.shape:", len(feature_list), feature_list[0].shape)
    
    utils.nansInData(feature_list, data_type="feature")
    
    # Save the features and the filelist order for later use.
    pickle.dump(feature_list, file=open((FEATURES_OUTPUT), mode = 'wb'))
    pickle.dump(X_test_paths, file = open((PATH_LIST), mode = 'wb'))
    
def main():
    featurize()


if __name__ == '__main__':
    main()
