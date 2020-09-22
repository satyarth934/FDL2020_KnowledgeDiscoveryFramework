





"""

USE THE NOTEBOOK FOR THIS TASK!!!


"""














from import_modules import *
sys.dont_write_bytecode = True

import utils


# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*" # <- needs to be normalized
DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*" # <- needs to be normalized
NORMALIZE = True

MODEL_NAME = "baseAE_orig_mean_median"
BASE_DIR = "/home/satyarth934/code/FDL_2020/"

OUTPUT_MODEL_PATH = BASE_DIR + "Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = BASE_DIR + "tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = BASE_DIR + "activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = BASE_DIR + "activation_viz/" + MODEL_NAME + "/train_test_paths.npy"
PATH_LIST = BASE_DIR + "Features/" + MODEL_NAME + "_filenames.pkl"
FEATURES_OUTPUT = BASE_DIR + "Features/" + MODEL_NAME + "_features.pkl"

NUM_EPOCHS = 200


# # Function to featurize the input
# def extract_features(img_array, model):
#             expanded_img_array = np.expand_dims(img_array, axis=0)
#             features = model.predict(expanded_img_array)
#             flattened_features = features.flatten()
#             normalized_features = flattened_features / np.linalg.norm(flattened_features)
#             return normalized_features


def similarity():
    X_test_paths = pickle.load(file=open((PATH_LIST), 'rb'))
    feature_list = pickle.load(file=open((FEATURES_OUTPUT), 'rb'))
    
    num_images = len(X_test_paths)
    num_features_per_image = len(feature_list[0])
    print("Number of images = ", num_images)
    print("Number of features per image = ", num_features_per_image)
    
    # Use scikit-learn to find Nearest Neighbors
    neighbors = NearestNeighbors(n_neighbors=50,
                                 algorithm='brute',
                                 metric='euclidean').fit(feature_list)
    
    
    
    
def main():
    similarity()


if __name__ == '__main__':
    main()
