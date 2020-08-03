#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import model
import utils
sys.dont_write_bytecode = True

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
# DATA_PATH = "/home/satyarth934/data/modis_data_products/*/array_3bands_normalized/448/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/mean_stdev_removed/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed/*"
# DATA_PATH = "/home/satyarth934/data/modis_data_products/terra/array_3bands_adapted/448/median_removed_gap_filled/*"
DATA_PATH = "/home/satyarth934/data/proxy_data/UCMerced_LandUse/Images/*/*"

NORMALIZE = True
MODEL_NAME = "baseAE_uc_merced"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200
BATCH_SIZE = 64
INTERPOLATE_DATA_GAP = False


def customGenerator(input_file_paths, dims, data_type):
    for i, file_path in enumerate(input_file_paths):
        if data_type.decode("utf-8") in ["png", "tif"]:
            img = plt.imread((file_path.decode("utf-8")))
        elif data_type.decode("utf-8") == "npy":
            img = np.load(file_path.decode("utf-8"))
        x = resize(img[:,:,:3], dims)
        
        yield x, x


def main():
    dims = (448, 448, 3)
    
    # Dataloader creation and test
    img_paths = glob.glob(DATA_PATH)
    print("len(img_paths):", len(img_paths))

    train_split = 0.6
    valid_split = 0.2
    test_split = 0.2
    X_train_paths = img_paths[:int(train_split * len(img_paths))]
    X_valid_paths = img_paths[int(train_split * len(img_paths)):int((train_split + valid_split) * len(img_paths))]
    X_test_paths = img_paths[len(img_paths) - int(test_split * len(img_paths)):]
    
    # More efficient data fetch pipeline
    AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
    
#     X_train = customGenerator(X_train_paths, dims)
#     X_test = customGenerator(X_test_paths, dims)
    train_dataset = tf.data.Dataset.from_generator(generator=customGenerator, output_types=(np.float32, np.float32), output_shapes=(dims, dims), args=[X_train_paths, dims, "tif"])
    output = list(train_dataset.take(1).as_numpy_iterator())

    print("@#@#@#@#@#@#@ 1")
    print("train_dataset:", train_dataset)
    print("len(output):", len(output))
    print("len(output[0]):", len(output[0]))
    x,y = output[0]
    print("Printing the output:", x.shape, y.shape)
    print("x:", x.min(), x.max())
    print("y:", y.min(), y.max())
    print(x.shape, y.shape)

    valid_dataset = tf.data.Dataset.from_generator(generator=customGenerator, output_types=(np.float32, np.float32), output_shapes=(dims, dims), args=[X_valid_paths, dims, "tif"])
    
    test_dataset = tf.data.Dataset.from_generator(generator=customGenerator, output_types=(tf.float32, tf.float32), args=[X_test_paths, dims, "tif"])
    
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
    test_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    
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
    complete_model = model.createModel(dims)
    print(complete_model.summary())

    complete_model.compile(optimizer='rmsprop', loss='mse')

    image = resize(plt.imread(X_test_paths[0])[:,:,:3], dims)
    print("Activation visualization image shape orig:", image.shape)
    image = np.expand_dims(image, 0)
    print("Activation visualization image shape extended:", image.shape)
    print("image:", image.min(), image.max())
#     image = X_test_reshaped[0:10]
#     print(image.shape)

    # Define the Activation Visualization callback
    output_dir = TENSORBOARD_LOG_DIR
    callbacks = [
        ActivationsVisualizationCallback(
            validation_data=(image,),
            layers_name=['conv2d_8'],
            output_dir=output_dir,
        ),
    ]

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf_callback")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)

    complete_model.fit(train_dataset,
                       epochs=NUM_EPOCHS,
                       steps_per_epoch=len(X_train_paths) // BATCH_SIZE,
#                        validation_split=0.2,
                       validation_data=valid_dataset,
                       validation_steps=len(X_valid_paths) // BATCH_SIZE,
                       callbacks=[callbacks, tensorboard_callback, WandbCallback()],
#                        use_multiprocessing=True
                       )

    complete_model.save(OUTPUT_MODEL_PATH)

#     sys.exit(0)


    # ## Model Testing

    # Define the Activation Visualization explainer
    # index = np.random.randint(0, len(X_test_reshaped))
    # image = input_test[index].reshape((1, 32, 32, 3))
    # image = np.expand_dims(X_test_reshaped[index],0)
#     image = X_test[0][0][:10]
    image = np.array([resize(plt.imread(test_img_path)[:,:,:3], dims) for test_img_path in X_test_paths[:10]])
#     image = X_test_reshaped[:10]
    label = image
    print('val:', image.shape)

    data = ([image])
    explainer = ExtractActivations()

    layers_of_interest = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7', 'conv2d_8', 'conv2d_transpose', 'conv2d_transpose_1', 'conv2d_transpose_2', 'conv2d_transpose_3', 'conv2d_9', 'conv2d_transpose_4']
    for layer in layers_of_interest:
        grid = explainer.explain(validation_data=data, model=complete_model, layers_name=[layer])
        print(grid.shape)
        explainer.save(grid, ACTIVATION_IMG_PATH, '%s.png' % (layer))


#     # complete_model = model.load_model(OUTPUT_MODEL_PATH)
#     for i in range(10):
#         index = np.random.randint(0, len(X_test_reshaped))

#         X_test_im = np.expand_dims(X_test_reshaped[index], 0)
#         out_image = np.squeeze(complete_model.predict(X_test_im))

#         im_min = out_image.min(axis=(0, 1), keepdims=True)
#         im_max = out_image.max(axis=(0, 1), keepdims=True)
#         out_image = (out_image - im_min) / (im_max - im_min)

#         print("Orig ", np.min(X_test_im), np.max(X_test_im))
#         print("Gen ", np.min(out_image), np.max(out_image))
#         fig = plt.figure()
#         plt.subplot(1, 3, 1)
#         plt.imshow(X_test_reshaped[index])
#         plt.subplot(1, 3, 2)
#         plt.imshow(np.squeeze(X_test_im))
#         plt.subplot(1, 3, 3)
#         plt.imshow(out_image)
#         plt.show()


if __name__ == "__main__":
    main()
