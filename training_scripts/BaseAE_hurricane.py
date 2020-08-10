#!/usr/bin/env python
# coding: utf-8

from import_modules import *
import model
import utils
sys.dont_write_bytecode = True

import wandb
from wandb.keras import WandbCallback
wandb.init(config={"hyper": "parameter"}, 
           name="baseAE_hurricane_ssim_featurizer_training", 
           notes="ssim training on hurricane, added early stopping and LR update if loss plateaus. Saves intermediate checkpoints.")

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
DATA_PATH = "/home/satyarth934/data/nasa_impact/hurricanes/*/*"

NORMALIZE = True
MODEL_NAME = "baseAE_hurricane_try3_ssim"
OUTPUT_MODEL_PATH = "/home/satyarth934/code/FDL_2020/Models/" + MODEL_NAME
TENSORBOARD_LOG_DIR = "/home/satyarth934/code/FDL_2020/tb_logs/" + MODEL_NAME
ACTIVATION_IMG_PATH = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME
PATH_LIST_LOCATION = "/home/satyarth934/code/FDL_2020/activation_viz/" + MODEL_NAME + "/train_test_paths.npy"

NUM_EPOCHS = 200
BATCH_SIZE = 64
INTERPOLATE_DATA_GAP = False


# Learning rate scheduler
def lr_scheduler_1(epoch, lr):
    decay_rate = 0.8
    decay_step = 1
    if epoch % decay_step == 0 and epoch:
        return lr * pow(decay_rate, np.floor(epoch / decay_step))
    return lr

def lr_scheduler(epoch, lr):
    decay_rate = 0.8
    decay_step = 5
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr


# Loss functtion
def ssim_loss(y_true, y_pred):
    loss=tf.reduce_mean(tf.image.ssim(y_true,y_pred,1.0,filter_size=3))
    return 1-loss


def ssim_loss_ms(y_true, y_pred):
    loss=tf.reduce_mean(tf.image.ssim_multiscale(y_true,y_pred,1.0,filter_size=3))
    return 1-loss


# Input data generation
def customGenerator(input_file_paths, dims, data_type):
    for i, file_path in enumerate(input_file_paths):
        if data_type.decode("utf-8") in ["png" or "tif"]:
            img = plt.imread((file_path.decode("utf-8")))
        elif data_type.decode("utf-8") == "npy":
            img = np.load(file_path.decode("utf-8"))
        x = resize(img[:,:,:3], dims)
            
        yield x, x

        
def main():
    dims = (448, 448, 3)
    
    # Dataloader creation and test
#     img_paths = glob.glob(DATA_PATH)
#     print("len(img_paths):", len(img_paths))
    
#     print("Discarding the unusable images")
#     img_paths = utils.getUsableImagePaths(img_paths, "png")
#     print("len(img_paths):", len(img_paths))
    f = open("hurricane_input.txt", 'r')
    img_paths = [fi.strip("\n") for fi in f.readlines()][:1024]
    print("len(img_paths):", len(img_paths))
    
    train_split = 1.0
    valid_split = 0.0
    test_split = 0.0
    X_train_paths = img_paths[:int(train_split * len(img_paths))]
    X_valid_paths = img_paths[int(train_split * len(img_paths)):int((train_split + valid_split) * len(img_paths))]
    X_test_paths = img_paths[len(img_paths) - int(test_split * len(img_paths)):]
    
    # More efficient data fetch pipeline
    AUTOTUNE = tensorflow.data.experimental.AUTOTUNE
    train_dataset = tf.data.Dataset.from_generator(generator=customGenerator, output_types=(np.float32, np.float32), output_shapes=(dims, dims), args=[X_train_paths, dims, "png"])
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

    valid_dataset = tf.data.Dataset.from_generator(generator=customGenerator, output_types=(np.float32, np.float32), output_shapes=(dims, dims), args=[X_valid_paths, dims, "png"])
    
    test_dataset = tf.data.Dataset.from_generator(generator=customGenerator, output_types=(tf.float32, tf.float32), args=[X_test_paths, dims, "png"])
    
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.cache().shuffle(buffer_size=3*BATCH_SIZE)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.cache().shuffle(buffer_size=3*BATCH_SIZE)
    valid_dataset = valid_dataset.repeat()
    valid_dataset = valid_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.map(utils.convert, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.cache()
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

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

    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    complete_model.compile(optimizer=opt, loss=ssim_loss)

    image = resize(plt.imread(X_train_paths[0])[:,:,:3], dims)
    print("Activation visualization image shape orig:", image.shape)
    image = np.expand_dims(image, 0)
    print("Activation visualization image shape extended:", image.shape)
    print("image:", image.min(), image.max())
#     image = X_test_reshaped[0:10]
#     print(image.shape)

    # Define the Activation Visualization callback
    output_dir = TENSORBOARD_LOG_DIR
    activation_viz_callbacks = [
        ActivationsVisualizationCallback(
            validation_data=(image,),
            layers_name=['conv2d_8'],
            output_dir=output_dir,
        ),
    ]

    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./tf_callback")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)

#     # Schedule LR updates
#     lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
    
    # Reduce LR only when the loss graph plateaus
    lr_plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.2,
                                                               patience=5, verbose=1, 
                                                               mode='auto', min_delta=0.0001, 
                                                               cooldown=5, min_lr=0.00000005)
    
    # save after every 10 epochs
    ckpt_path = OUTPUT_MODEL_PATH + "/checkpoints/"
    subprocess.call("mkdir -p %s" % ckpt_path, shell=True)
    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=ckpt_path + "epoch{epoch:03d}-loss{loss:.6f}.hdf5",
        monitor='loss', verbose=1, save_best_only=False,
        save_weights_only=False, mode='auto',
        save_freq=(10 * (len(X_train_paths) // BATCH_SIZE)))

    earlystopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='loss', min_delta=0.000001, patience=5, verbose=1, mode='auto')
    
    callback_functions = [activation_viz_callbacks, 
                          tensorboard_callback, 
                          lr_plateau_callback, 
                          model_ckpt_callback,
                          earlystopping_callback,
                          WandbCallback()]
    
    model_history = complete_model.fit(train_dataset,
                                       epochs=NUM_EPOCHS,
                                       steps_per_epoch=len(X_train_paths) // BATCH_SIZE,
                                       initial_epoch=0,
#                                        validation_split=0.2,
#                                        validation_data=valid_dataset,
#                                        validation_steps=len(X_valid_paths) // BATCH_SIZE,
                                       callbacks=callback_functions,
#                                        use_multiprocessing=True
                                      )
        
#     print("model_history.epoch:", model_history.epoch)
#     print("model_history:")
#     pprint(model_history.history)
    
    complete_model.save(OUTPUT_MODEL_PATH)

#     sys.exit(0)


    # ## Model Testing

    # Define the Activation Visualization explainer
    # index = np.random.randint(0, len(X_test_reshaped))
    # image = input_test[index].reshape((1, 32, 32, 3))
    # image = np.expand_dims(X_test_reshaped[index],0)
#     image = X_test[0][0][:10]
    image = np.array([resize(plt.imread(train_img_path)[:,:,:3], dims) for train_img_path in X_train_paths[:10]])
#     image = np.array([resize(plt.imread(test_img_path)[:,:,:3], dims) for test_img_path in X_test_paths[:10]])
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


    # complete_model = model.load_model(OUTPUT_MODEL_PATH)
    for i in range(10):
        index = np.random.randint(0, len(X_train_paths))

        image_train = resize(plt.imread(X_train_paths[index])[:,:,:3], dims)
        X_image_train = np.expand_dims(image_train, 0)
        out_image = np.squeeze(complete_model.predict(X_image_train))

        im_min = out_image.min(axis=(0, 1), keepdims=True)
        im_max = out_image.max(axis=(0, 1), keepdims=True)
        out_image = (out_image - im_min) / (im_max - im_min)

        print("Orig ", np.min(X_image_train), np.max(X_image_train))
        print("Gen ", np.min(out_image), np.max(out_image))
#         fig = plt.figure()
#         plt.subplot(1, 3, 1)
#         plt.imshow(image_train)
#         plt.subplot(1, 3, 2)
#         plt.imshow(np.squeeze(X_image_train))
#         plt.subplot(1, 3, 3)
#         plt.imshow(out_image)
#         plt.show()


if __name__ == "__main__":
    main()
