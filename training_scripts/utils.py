from import_modules import *
sys.dont_write_bytecode = True


def normalize(mat):
    valid_cells = np.invert(np.isnan(mat))
    normalized = np.subtract(mat, np.nanmin(mat), where=valid_cells) / (np.nanmax(mat) - np.nanmin(mat))
    return normalized


def convert(image, label):
    image = tensorflow.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]
    label = tensorflow.image.convert_image_dtype(label, tf.float32)
    return image, label


