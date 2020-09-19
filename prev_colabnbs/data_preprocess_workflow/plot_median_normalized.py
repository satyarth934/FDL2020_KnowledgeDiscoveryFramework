# Import packages
import numpy as np
from matplotlib import pyplot as plt

# Functions
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    return (array - array_min) / (array_max - array_min)

# Main
data_folder = '/media/francesco/passport/fdl/modis_output_terra/array_med_mean_stdev/'
output = '/media/francesco/passport/fdl/modis_output_terra/array_med_mean_stdev/'

pixel_lengths = [224, 448, 672]

for pixel_length in pixel_lengths:
    band1 = f'{data_folder}{pixel_length}_band_1_stdev.npy'
    band4 = f'{data_folder}{pixel_length}_band_4_stdev.npy'
    band3 = f'{data_folder}{pixel_length}_band_3_stdev.npy'

    band1_data = np.load(band1)
    band4_data = np.load(band4)
    band3_data = np.load(band3)

    band1_normalized = normalize(band1_data)
    band4_normalized = normalize(band4_data)
    band3_normalized = normalize(band3_data)

    band_stacked = np.stack((band1_normalized, band4_normalized, band3_normalized))
    band_stacked = np.swapaxes(band_stacked, axis1=0, axis2=1)
    band_stacked = np.swapaxes(band_stacked, axis1=1, axis2=2)

    plt.imshow(band_stacked)

    plt.savefig(f'{output}{pixel_length}_stdev.png')

    cake = 'good'
