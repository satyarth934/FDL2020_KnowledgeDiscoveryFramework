# -*- coding: utf-8 -*-
'''
Utility files for data modification

Sources:
1. tensorflow/models/research/inception/inception/data/build_image_data.py

'''
import os
import numpy as np
import tensorflow as tf
import glob
import random
import time
#from multiprocessing import Process

# tf.enable_eager_execution()

RANDOM_SEED=42

RECORD_FILE_NUM=0
TRAIN_RATIO=0.80
VALID_RATIO=0.10
NUM_IMAGES=1000
NUM=0
DIM=(400,400,3)
random.seed(42)

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def to_tfrecords(raw_directory,image_globs,label_globs,split):
    '''
    Take in a directory consisting of classes/images, split into train-valid-test
    if necessary and convert into tfrecord set.
    '''
    save_directory=raw_directory+"tfrecords/"+split+'/'
    for i, img_path in enumerate(image_globs):
        #print(img_path)
        tfrecord_fname="{}.tfrecords-{}".format(split,i)
        writer=tf.io.TFRecordWriter(save_directory+tfrecord_fname)
        # print("Creating the %.3d tfrecord file"%i)

        img=np.load(img_path).tobytes()
        data={'image': _bytes_feature(img),
              'label': _bytes_feature(bytes(label_globs[i],'utf-8'))}

        feature = tf.train.Features(feature=data)  # Wrap the data as TensorFlow Features.
        example = tf.train.Example(features=feature)  # Wrap again as a TensorFlow Example.
        serialized = example.SerializeToString()  # Serialize the data.
        writer.write(serialized)  # Write the serialized data to the TFRecords file.

def parser(proto,shape=(400,400,3)):
    data={'image': tf.io.FixedLenFeature([],tf.string)}
    parsed_example=tf.io.parse_single_example(serialized=proto,features=data)
    image_raw=parsed_example['image']
    image=tf.io.decode_raw(image_raw,tf.float32)
    image=tf.reshape(image,shape=shape)
    return image,image # Since input image==label making a copy (might be inefficient)

def from_tfrecords(records_globs,split,batch_size):
    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False
    AUTO=tf.data.experimental.AUTOTUNE

    dataset=tf.data.TFRecordDataset(records_globs, num_parallel_reads=AUTO)
    dataset=dataset.with_options(option_no_order)
    dataset=dataset.interleave(tf.data.TFRecordDataset, cycle_length=16, num_parallel_calls=AUTO)

    dataset=dataset.map(parser, num_parallel_calls=AUTO)
    dataset=dataset.cache()
    if split=="train":
        dataset=dataset.shuffle(buffer_size=1000).repeat().batch(batch_size)
    else:
        dataset=dataset.batch(batch_size)
    dataset=dataset.prefetch(buffer_size=AUTO)

    #return tf.contrib.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    return dataset





if __name__=="__main__":
    raw_directory="Datasets/MODIS_MCD43A4/"
    image_directory=raw_directory+"Globe/training_set/"


    image_globs=glob.glob(image_directory+"*/np_arrays/*.npy")
    label_globs=[p.split('/')[-1].split('.npy')[0] for p in image_globs]

    len_globs=len(image_globs)
    dataset=list(zip(image_globs,label_globs))
    random.shuffle(dataset)

    image_globs,label_globs=zip(*dataset)

    train_index=int(TRAIN_RATIO*len_globs)
    valid_index=train_index+int(VALID_RATIO*len_globs)
    test_index=valid_index+int((1-TRAIN_RATIO-VALID_RATIO)*len_globs)

    train_globs=image_globs[0:train_index]
    valid_globs=image_globs[train_index:valid_index]
    test_globs=image_globs[valid_index:test_index]

    train_labels=label_globs[0:train_index]
    valid_labels=label_globs[train_index:valid_index]
    test_labels=label_globs[valid_index:test_index]

    init_t=time.time()
    to_tfrecords(raw_directory=raw_directory,
                image_globs=train_globs,
                label_globs=train_labels,
                split="train")
    train_t=time.time()
    print("Train done ",train_t-init_t)
    to_tfrecords(raw_directory=raw_directory,
                image_globs=valid_globs,
                label_globs=valid_labels,
                split="valid")
    valid_t=time.time()
    print("Valid done ",valid_t-train_t)
    to_tfrecords(raw_directory=raw_directory,
                image_globs=test_globs,
                label_globs=test_labels,
                split="test")
    test_t=time.time()
    print("Test done ",test_t-valid_t)
