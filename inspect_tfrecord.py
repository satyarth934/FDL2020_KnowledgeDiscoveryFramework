import tensorflow as tf


def main():
    tfr_file = "/home/satyarth934/Projects/NASA_FDL_2020/Datasets/MODIS_MCD43A4/tfrecords/train/train.tfrecords-1"

    for example in tf.compat.v1.python_io.tf_record_iterator(tfr_file):
        print(tf.train.Example.FromString(example))


if __name__ == '__main__':
    main()
