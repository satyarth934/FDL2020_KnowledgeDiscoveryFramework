from import_modules import *
sys.dont_write_bytecode = True

import utils


# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*" # <- needs to be normalized
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*" # <- needs to be normalized
DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed_gap_filled/*"
NORMALIZE = True

MODEL_NAME = "baseAE_median_localRandom_in_swatch"
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


# Function to featurize the input
def extract_features(img_array, model, layer_names):
    outputs = [layer.output for layer in model.layers if layer.name in layer_names]          # all layer outputs
#     print(len(outputs))
    
    functor = K.function([model.input], outputs)   # evaluation function
#     print(functor)
    
    # Testing
    layer_outs = functor([img_array])
#     print(len(layer_outs), layer_outs[0].shape)
    feature_list = layer_outs[0]
    
    flat_features = [f.flatten() for f in feature_list]
    normed_features = [f/np.linalg.norm(f) for f in flat_features]
    
    return normed_features


def featurize():
    print("---- Reading Data ----")
    img_paths = glob.glob(DATA_PATH)

    print("len(img_paths):", len(img_paths))
    random.seed(a=13521)
    random.shuffle(img_paths)

    train_test_split = 0.8
    X_test_paths = img_paths[int(train_test_split * len(img_paths)):]

    dims = (448, 448, 3)

    # Loading Data
    X_test = utils.getData(X_test_paths, dims)
    print("X_test:", X_test.shape)

    # To check NaN pixel images
    nan_pixels_per_image = utils.nansInData(X_test)
    # plt.scatter(x=np.arange(0,len(nan_pixels_per_image)), y=nan_pixels_per_image)
    # plt.savefig("nan_scatter.png")

    # Checking min max to see if normalization is needed or not
    print("Before normalization")
    print(np.nanmin(X_test), np.nanmax(X_test))

    X_test = utils.normalize(X_test)

    # Checking min max after normalization
    print("After normalization")
    print(np.nanmin(X_test), np.nanmax(X_test))

    # Interpolate nan values
    X_test = utils.interpolateNaNValues(X_test)

    # To check NaN pixel images
    nan_pixels_per_image = utils.nansInData(X_test)

    print("---- Reading Model ----")
    model = load_model(OUTPUT_MODEL_PATH)
    print(model.summary())
    
    print("---- Featurizing Data ----")
    feature_list = extract_features(img_array=X_test, model=model, layer_names=['conv2d_8'])

#     layer_name = 'conv2d_8'
#     intermediate_layer_model = Model(inputs=model.input,
#                                      outputs=model.get_layer(layer_name).output)
#     intermediate_output = intermediate_layer_model.predict(data)
#     feature_list = intermediate_output
    
    utils.nansInData(feature_list, data_type="feature")
    
    # Save the features and the filelist order for later use.
    pickle.dump(feature_list, file=open((FEATURES_OUTPUT), mode = 'wb'))
    pickle.dump(X_test_paths, file = open((PATH_LIST), mode = 'wb'))
    
def main():
    featurize()


if __name__ == '__main__':
    main()
