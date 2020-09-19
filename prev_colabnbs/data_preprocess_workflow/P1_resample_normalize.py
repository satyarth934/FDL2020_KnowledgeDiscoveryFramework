"""
This code takes in the 3-channel array created in P0 and processes it by:
- Rescaling the array to 448x448 and 672x672
- Interpolating bad pixels
- Normalizing the data (0 and 1)

"""

# Import packages
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from matplotlib import cm
import datetime
import glob
from joblib import Parallel, delayed


# Setup functions
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    return (array - array_min) / (array_max - array_min)


def resize_norm(input_basename, input_data, output_side_pixel, interp_style, save_loc):
    """
    Uses the open-cv library to downscale the data

    Parameters
    ----------
    input_basename : [str] Input filename that will be used to save the images and arrays
    input_data : [np array] Input 3-channel data
    output_side_pixel : [int] Length of one side of the output image (in pixels)
    interp_style : [str] Interpolation used
    save_loc : [str] Output save location

    Returns
    -------

    """
    # Get instrument information from the basename
    if input_basename[6:9] == 'MOD':
        instrument_name = 'Terra'
    else:
        instrument_name = 'Aqua'

    # Get the date from the filename
    jdate = f'{input_basename.split(".")[1][1:5]}-{input_basename.split(".")[1][5:8]}'

    # Convert from Julian day because it's not easy
    fmt = "%Y-%j"
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    datestr = f'{datestd.year}-{datestd.month}-{datestd.day}'

    # Downscale the array and remove nans
    downscaled_array = cv2.resize(input_data, dsize=(output_side_pixel, output_side_pixel), interpolation=interp_style)

    # Normalize the data
    band1_norm = normalize(downscaled_array[:, :, 0])
    band4_norm = normalize(downscaled_array[:, :, 1])
    band3_norm = normalize(downscaled_array[:, :, 2])
    normalized_3stack = np.stack((band1_norm, band4_norm, band3_norm))
    normalized_3stack = np.swapaxes(normalized_3stack, axis1=0, axis2=1)
    normalized_3stack = np.swapaxes(normalized_3stack, axis1=1, axis2=2)

    # Save the data into a new numpy array
    np.save(f'{save_loc}array_3bands_normalized/{output_side_pixel}/'
            f'{input_basename}_norm_{output_side_pixel}.npy', normalized_3stack, allow_pickle=True)

    # Plot the result
    fig, axs = plt.subplots(2, 2, figsize=(8, 12))
    axs[0, 0].imshow(band1_norm[:, :], cmap=cm.jet)
    axs[0, 0].set(ylabel='Pixel Latitude')
    axs[0, 0].set(xlabel='Pixel Longitude')
    axs[0, 0].set(title='Band 1 (620-670 nm)')

    axs[0, 1].imshow(band4_norm[:, :], cmap=cm.jet)
    axs[0, 1].set(ylabel='Pixel Latitude')
    axs[0, 1].set(xlabel='Pixel Longitude')
    axs[0, 1].set(title='Band 4 (545-565 nm)')

    axs[1, 0].imshow(band3_norm[:, :], cmap=cm.jet)
    axs[1, 0].set(ylabel='Pixel Latitude')
    axs[1, 0].set(xlabel='Pixel Longitude')
    axs[1, 0].set(title='Band 3 (459-479 nm)')

    axs[1, 1].imshow(normalized_3stack, cmap=cm.jet)
    axs[1, 1].set(ylabel='Pixel Latitude')
    axs[1, 1].set(xlabel='Pixel Longitude')
    axs[1, 1].set(title='Full color')

    fig.suptitle(f'{instrument_name} : {datestr} ({jdate})', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{save_loc}array_3bands_normalized_images/{output_side_pixel}/'
                f'{input_basename}_norm_{output_side_pixel}.png')
    plt.close(fig)

    return


def processing_wrapper(input_numpy_file, input_band_sizes, out_folder):
    """

    Parameters
    ----------
    input_numpy_file : [str] Input 3-channel numpy file created in P0
    input_band_sizes : [array] Resize length of images in pixels
    out_folder : [str] output folder

    Returns
    -------

    """
    # Get the filename and the instrument name
    basename = os.path.basename(input_numpy_file)[0:-4]

    # Load the data and set the default bad values to np.nan
    data = np.load(input_numpy_file)
    nan_indices = np.where(data == -28672)
    data_float = data.astype(float)
    data_float[nan_indices] = np.nan

    # Downsize the data and normalize each band
    for band_size in input_band_sizes:
        resize_norm(basename, data_float, band_size, cv2.INTER_CUBIC, out_folder)

    print(f'Finished processing {basename}...')

    return


# Main
input_folder = '/media/francesco/passport/fdl/modis_output_aqua/array_3bands/'
output_folder = '/media/francesco/passport/fdl/modis_output_aqua/'

# Select the length of the length/widths of the output images in pixels
output_widths = [224, 448, 672]

# Need to create the output folders before parallelization to prevent bugs
if not os.path.exists(f'{output_folder}array_3bands_normalized/'):
    os.mkdir(f'{output_folder}array_3bands_normalized/')
if not os.path.exists(f'{output_folder}array_3bands_normalized_images/'):
    os.mkdir(f'{output_folder}array_3bands_normalized_images/')
for output_width in output_widths:
    if not os.path.exists(f'{output_folder}array_3bands_normalized_images/{output_width}'):
        os.mkdir(f'{output_folder}array_3bands_normalized_images/{output_width}')
    if not os.path.exists(f'{output_folder}array_3bands_normalized/{output_width}'):
        os.mkdir(f'{output_folder}array_3bands_normalized/{output_width}')

# Get a list of files to use
datafiles = glob.glob(f'{input_folder}*.npy')

# Run the processing
# Single-core
# for datafile in datafiles:
#     processing_wrapper(datafile, output_widths, output_folder)

# Multi-core
num_cores = 4
Parallel(n_jobs=num_cores)(delayed(processing_wrapper)(datafile, output_widths, output_folder)
                           for datafile in datafiles)
