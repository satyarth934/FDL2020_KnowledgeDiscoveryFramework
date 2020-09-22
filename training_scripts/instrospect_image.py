#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import model
import utils


"""
Parameters:
1. Data path
2. Normalize or not
3. Tensorboard Log Directory
4. Model path to save
"""
# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*"
DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*"
NORMALIZE = True
MODEL_NAME = "baseAE_median_median_in_swatch"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200


def main():
    img_paths = glob.glob(DATA_PATH)
    print("len(img_paths):", len(img_paths))
    
    X_train_paths = img_paths
    
    dims = (448, 448, 3)

    # Loading Data
    X_train = utils.getData(X_train_paths, dims)
    print("X_train:", X_train.shape)

    # To check NaN pixel images
    nan_pixels_per_image = utils.nansInData(X_train)
    # plt.scatter(x=np.arange(0,len(nan_pixels_per_image)), y=nan_pixels_per_image)
    # plt.savefig("nan_scatter.png")

    print("Usable images:", np.sum(np.array(nan_pixels_per_image) < 20000))
    
    print("Before normalization")
    print(np.nanmin(X_train), np.nanmax(X_train))
    
    X_train = utils.normalize(X_train)

    # Checking min max after normalization
    print("After normalization")
    print(np.nanmin(X_train), np.nanmax(X_train))
    
    # Put median instead of black strip
    median_img = utils.getMedian()
    for i in range(len(X_train)):
        X_train[i,:,:,:][np.isnan(X_train[i,:,:,:])] = median_img[np.isnan(X_train[i,:,:,:])]
    
#     plt.imshow(median_img); plt.savefig("median_img")
    
    nan_pixels_per_image = utils.nansInData(X_train)
    print("Usable images:", np.sum(np.array(nan_pixels_per_image) < 20000))


if __name__ == "__main__":
    main()