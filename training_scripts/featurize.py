from import_modules import *
sys.dont_write_bytecode = True


DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*" # <- needs to be normalized
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*" # <- needs to be normalized
NORMALIZE = True

MODEL_NAME = "baseAE_orig"
BASE_DIR = "/home/satyarth934/code/FDL_2020/"

OUTPUT_MODEL_PATH = BASE_DIR + "Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = BASE_DIR + "tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = BASE_DIR + "activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = BASE_DIR + "activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200


def featurize():
	# NEW MODIS DATASET

	img_paths = glob.glob(DATA_PATH)
	print("len(img_paths):", len(img_paths))
	random.seed(a=13521)
	random.shuffle(img_paths)

	train_test_split = 0.8
	X_test_paths = img_paths[int(train_test_split * len(img_paths)):]

	dims = (448, 448, 3)

	# Loading Testing Data
	X_test = np.empty((len(X_test_paths), *dims))
	for i, p in enumerate(X_test_paths):
	    X_test[i, :, :, :] = np.load(p)

	print("X_test:", X_test.shape)

	# To check what percentage of pixels are 'nan'
	print(np.sum(np.isnan(X_test)) / np.prod(X_test.shape))

	# Checking min max to see if normalization is needed or not
	print("Before normalization")
	print(np.nanmin(X_test), np.nanmax(X_test))

	# Normalize Inputs
	def normalize(mat):
	    valid_cells = np.invert(np.isnan(mat))
	    normalized = np.subtract(mat, np.nanmin(mat), where=valid_cells) / (np.nanmax(mat) - np.nanmin(mat))
	    return normalized

	X_test = normalize(X_test)

	# Checking min max after normalization
	print("After normalization")
	print(np.nanmin(X_test), np.nanmax(X_test))

	# Function to featurize the input
	def extract_features(img_array, model):
	    expanded_img_array = np.expand_dims(img_array, axis=0)
	    features = model.predict(expanded_img_array)
	    flattened_features = features.flatten()
	    normalized_features = flattened_features / np.linalg.norm(flattened_features)
	    return normalized_features

	# Featurize all input images
	feature_list = []
	for i in tqdm_notebook(range(len(X_test))):
	    feature_list.append(extract_features(X_test[i, :, :, :], model))


def main():
	featurize()


if __name__ == '__main__':
	main()
