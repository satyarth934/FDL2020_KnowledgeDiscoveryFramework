import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import glob
import numpy as np
# import IPython.display as display
import matplotlib.pyplot as plt
from io import BytesIO
tf_globspath = '/home/satyarth934/Projects/NASA_FDL_2020/Datasets/MODIS_MCD43A4/tfrecords/train/train*'
tf_globs = glob.glob(tf_globspath)
raw_dataset = tf.data.TFRecordDataset(tf_globs)
# Create a dictionary describing the features.
image_feature_description = {
    'label': tf.io.FixedLenFeature([], tf.string),
    'image': tf.io.FixedLenFeature([], tf.string),
}


def _parse_image_function(example_proto):
  # Parse the input tf.Example proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, image_feature_description)


parsed_image_dataset = raw_dataset.map(_parse_image_function)
print(parsed_image_dataset)
for image_features in parsed_image_dataset:
    im = image_features['image']
    im = tf.io.decode_raw(im, out_type=float).numpy()
    im = im.reshape((400, 400, 3))
    # >>> loaded_np = np.load(load_bytes, allow_pickle=True)
    plt.imshow(im)
    plt.show()
