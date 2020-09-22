from import_modules import *
sys.dont_write_bytecode = True


def normalize(mat):
#     valid_cells = np.invert(np.isnan(mat))
#     normalized = np.subtract(mat, np.nanmin(mat), where=valid_cells) / (np.nanmax(mat) - np.nanmin(mat))
    normalized = (mat - np.nanmin(mat)) / (np.nanmax(mat) - np.nanmin(mat))
    return normalized


def convert(image, label):
    image = tensorflow.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
    label = tensorflow.image.convert_image_dtype(label, tf.float32)
    return image, label


def getMedian():
    median_r_path = "/home/satyarth934/data/modis_data_products/terra/448_band_1_median.npy"
    median_g_path = "/home/satyarth934/data/modis_data_products/terra/448_band_4_median.npy"
    median_b_path = "/home/satyarth934/data/modis_data_products/terra/448_band_3_median.npy"

    median_r = np.load(median_r_path)
    median_g = np.load(median_g_path)
    median_b = np.load(median_b_path)

    median = np.dstack([median_r, median_g, median_b])
    
    return median


def getData(paths, dims, file_type="npy"):
    loaded_data = np.empty((len(paths), *dims))
    if file_type == "npy":
        for i, p in enumerate(tqdm(paths)):
            loaded_data[i, :, :, :] = np.load(p)
    elif file_type in ["png", "jpeg", "jpg"]:
        for i, p in enumerate(tqdm(paths)):
            loaded_data[i, :, :, :] = resize(plt.imread(p)[:,:,:3], dims)
    
    return loaded_data


def nansInData(dataset, data_type="image"):
    nan_pixels_per_image = []
    if data_type == "image":
        for i in range(len(dataset)):
            nan_pixels_per_image.append(np.sum(np.isnan(dataset[i,:,:,:])))
    elif data_type == "feature":
        for i in range(len(dataset)):
            nan_pixels_per_image.append(np.sum(np.isnan(dataset[i])))
    
    print("Total NaN pix images:", np.sum(np.array(nan_pixels_per_image) > 0))
    
    return nan_pixels_per_image


# fill in the nan values in the dataset
def interpolateNaNValues(dataset):
#     # METHOD 1: completely random values ------------------------------------
#     dataset[np.isnan(dataset)] = np.random.random(np.sum(np.isnan(dataset)))

    # METHOD 2: Set nan values to median image values -----------------------
    median_img = getMedian()
    for i in range(len(dataset)):
        dataset[i,:,:,:][np.isnan(dataset[i,:,:,:])] = median_img[np.isnan(dataset[i,:,:,:])]
    plt.imshow(median_img); plt.savefig("median_img.png")

    # METHOD 3: Replace with the closest N randomly chosen points in the vicinity of the nan pixels -----
    # Check "fill_data_gaps.py"

    return dataset


def getUsableImagePaths(image_paths, data_type):
    usable_paths = []
    for file_path in tqdm(image_paths):
        if data_type in ["png" or "tif"]:
            img = plt.imread((file_path))
        elif data_type == "npy":
            img = np.load(file_path)
        
        # If more than 30% if the image is nan values (USELESS)
        if (np.sum((np.isnan(img))) / np.product(img.shape)) > 0.3:
            continue
        
        # If the entire image is just a single value (USELESS)
        if np.nanmin(img) == np.nanmax(img):
            continue
        
        usable_paths.append(file_path)
    return usable_paths


# prepare a dataset with the specified number of samples per class
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


# if __name__=="__main__":
#     DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*"
    
#     img_paths = glob.glob(DATA_PATH)
#     train_test_split = 0.8
#     X_test_paths = img_paths[int(train_test_split * len(img_paths)):]

#     dims = (448, 448, 3)

#     # Loading Data
#     X_test = getData(X_test_paths, dims)
#     X_test = normalize(X_test)

    