"""
Read in the HDF Level 2 data from MODIS 500 m and output PNGs

FC 7/22/2020
"""

# Import packages
from pyhdf.SD import SD, SDC
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import glob
import os
import datetime
from joblib import Parallel, delayed


# Setup functions
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    return (array - array_min) / (array_max - array_min)


def process_hdf(input_hdf, outdir):
    """
    Processes MODIS Level 9 data

    Parameters
    ----------
    input_hdf : [str] Input hdf file
    outdir : [str] Output directory of figures and numpy files

    Returns none
    -------

    """

    # Isolate the basename of the file
    basename = input_hdf.split('/')[-1]
    outname = f'{basename[0:-4]}'

    # Check if the output file is already there. If so, skip it
    if os.path.exists(f'{outdir}array_3bands/3band_{outname}.npy'):
        print(f'3band_{outname} already done. Skipping...')
        return

    # Find the instrument name from the filename
    if basename[0:3] == 'MOD':
        instrument_name = 'Terra'
    else:
        instrument_name = 'Aqua'

    # Find the date from the filename
    jdate = f'{basename.split(".")[1][1:5]}-{basename.split(".")[1][5:8]}'

    # Convert from Julian day because it's not easy
    fmt = "%Y-%j"
    datestd = datetime.datetime.strptime(jdate, fmt).date()
    datestr = f'{datestd.year}-{datestd.month}-{datestd.day}'

    # Read in the hdf file
    # First take the 200m-500m aggregate
    hdf = SD(input_hdf, SDC.READ)

    # Now the 500m reflective solar bands
    band1_ref = hdf.select('sur_refl_b01_1')
    band1 = band1_ref[:, :]
    band1_nan_indices = np.where(band1 == -28672)
    band1_mia_pixels = len(band1_nan_indices[0])

    band2_ref = hdf.select('sur_refl_b02_1')
    band2 = band2_ref[:, :]
    band2_nan_indices = np.where(band2 == -28672)
    band2_mia_pixels = len(band2_nan_indices[0])

    band3_ref = hdf.select('sur_refl_b03_1')
    band3 = band3_ref[:, :]
    band3_nan_indices = np.where(band3 == -28672)
    band3_mia_pixels = len(band3_nan_indices[0])

    band4_ref = hdf.select('sur_refl_b04_1')
    band4 = band4_ref[:, :]
    band4_nan_indices = np.where(band4 == -28672)
    band4_mia_pixels = len(band4_nan_indices[0])

    band5_ref = hdf.select('sur_refl_b05_1')
    band5 = band5_ref[:, :]
    band5_nan_indices = np.where(band5 == -28672)
    band5_mia_pixels = len(band5_nan_indices[0])

    band6_ref = hdf.select('sur_refl_b06_1')
    band6 = band6_ref[:, :]
    band6_nan_indices = np.where(band6 == -28672)
    band6_mia_pixels = len(band6_nan_indices[0])

    band7_ref = hdf.select('sur_refl_b07_1')
    band7 = band7_ref[:, :]
    band7_nan_indices = np.where(band7 == -28672)
    band7_mia_pixels = len(band7_nan_indices[0])

    # Save the number of pixels that are missing for the file
    mia_pixels = [band1_mia_pixels, band2_mia_pixels, band3_mia_pixels, band4_mia_pixels,
                  band5_mia_pixels, band6_mia_pixels, band7_mia_pixels]
    np.save(f'{outdir}log_files/{outname}_mia_pixels.npy', mia_pixels, allow_pickle=True)

    # Create numpy arrays with the data and save them
    array_3stack = np.stack((band1, band4, band3))
    array_7stack = np.stack((band1, band2, band3, band4,
                             band5, band6, band7))

    # Shift the columns around so it's lat, lon, channel
    array_3stack = np.swapaxes(array_3stack, axis1=0, axis2=1)
    array_3stack = np.swapaxes(array_3stack, axis1=1, axis2=2)
    array_7stack = np.swapaxes(array_7stack, axis1=0, axis2=1)
    array_7stack = np.swapaxes(array_7stack, axis1=1, axis2=2)

    # Save data here
    np.save(f'{outdir}array_3bands/3band_{outname}.npy', array_3stack, allow_pickle=True)
    np.save(f'{outdir}array_7bands/7band_{outname}.npy', array_7stack, allow_pickle=True)

    # For plotting, we have to convert the missing values to nan and convert the data-type to float
    total_nans_3stack = np.where(array_3stack == -28672)
    array_3stack_float = array_3stack.astype(float)
    array_3stack_float[total_nans_3stack] = np.nan
    total_nans = np.where(array_7stack == -28672)
    array_7stack_float = array_7stack.astype(float)
    array_7stack_float[total_nans] = np.nan

    # Plot 1 - Original values
    fig1, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs[0, 0].imshow(array_7stack_float[:, :, 0], cmap=cm.jet)
    axs[0, 0].set(ylabel='Pixel Latitude')
    axs[0, 0].set(xlabel='Pixel Longitude')
    axs[0, 0].set(title='Band 1 (620-670 nm)')

    axs[0, 1].imshow(array_7stack_float[:, :, 1], cmap=cm.jet)
    axs[0, 1].set(title='Band 2 (841-876 nm)')
    # axs[0, 1].set(ylabel='Pixel Latitude')
    axs[0, 1].set(xlabel='Pixel Longitude')

    axs[0, 2].imshow(array_7stack_float[:, :, 2], cmap=cm.jet)
    # axs[0, 2].set(ylabel='Pixel Latitude')
    axs[0, 2].set(xlabel='Pixel Longitude')
    axs[0, 2].set(title='Band 3 (459-479 nm)')

    axs[0, 3].imshow(array_7stack_float[:, :, 3], cmap=cm.jet)
    # axs[0, 3].set(ylabel='Pixel Latitude')
    axs[0, 3].set(xlabel='Pixel Longitude')
    axs[0, 3].set(title='Band 4 (545-565 nm)')

    axs[1, 0].imshow(array_7stack_float[:, :, 4], cmap=cm.jet)
    axs[1, 0].set(ylabel='Pixel Latitude')
    axs[1, 0].set(xlabel='Pixel Longitude')
    axs[1, 0].set(title='Band 5 (1230-1250 nm)')

    axs[1, 1].imshow(array_7stack_float[:, :, 5], cmap=cm.jet)
    # axs[1, 1].set(ylabel='Pixel Latitude')
    axs[1, 1].set(xlabel='Pixel Longitude')
    axs[1, 1].set(title='Band 6 (1628-1652 nm)')

    axs[1, 2].imshow(array_7stack_float[:, :, 6], cmap=cm.jet)
    # axs[1, 2].set(ylabel='Pixel Latitude')
    axs[1, 2].set(xlabel='Pixel Longitude')
    axs[1, 2].set(title='Band 7 (2105-2155 nm)')
    fig1.delaxes(axs[1, 3])
    fig1.suptitle(f'{instrument_name} : {datestr} ({jdate})', fontsize=16, fontweight='bold')
    fig1.tight_layout()
    plt.savefig(f'{outdir}bands_imaged/ORIGINAL_{outname}.png')
    plt.close(fig1)

    # Plot 1 - Vmax changed
    fig2, axs = plt.subplots(2, 4, figsize=(15, 10))
    axs[0, 0].imshow(array_7stack_float[:, :, 0], cmap=cm.jet, vmax=3e3)
    axs[0, 0].set(ylabel='Pixel Latitude')
    axs[0, 0].set(xlabel='Pixel Longitude')
    axs[0, 0].set(title='Band 1 (620-670 nm)')

    axs[0, 1].imshow(array_7stack_float[:, :, 1], cmap=cm.jet, vmax=3e3)
    axs[0, 1].set(title='Band 2 (841-876 nm)')
    # axs[0, 1].set(ylabel='Pixel Latitude')
    axs[0, 1].set(xlabel='Pixel Longitude')

    axs[0, 2].imshow(array_7stack_float[:, :, 2], cmap=cm.jet, vmax=3e3)
    # axs[0, 2].set(ylabel='Pixel Latitude')
    axs[0, 2].set(xlabel='Pixel Longitude')
    axs[0, 2].set(title='Band 3 (459-479 nm)')

    axs[0, 3].imshow(array_7stack_float[:, :, 3], cmap=cm.jet, vmax=3e3)
    # axs[0, 3].set(ylabel='Pixel Latitude')
    axs[0, 3].set(xlabel='Pixel Longitude')
    axs[0, 3].set(title='Band 4 (545-565 nm)')

    axs[1, 0].imshow(array_7stack_float[:, :, 4], cmap=cm.jet, vmax=3e3)
    axs[1, 0].set(ylabel='Pixel Latitude')
    axs[1, 0].set(xlabel='Pixel Longitude')
    axs[1, 0].set(title='Band 5 (1230-1250 nm)')

    axs[1, 1].imshow(array_7stack_float[:, :, 5], cmap=cm.jet, vmax=3e3)
    # axs[1, 1].set(ylabel='Pixel Latitude')
    axs[1, 1].set(xlabel='Pixel Longitude')
    axs[1, 1].set(title='Band 6 (1628-1652 nm)')

    axs[1, 2].imshow(array_7stack_float[:, :, 6], cmap=cm.jet, vmax=3e3)
    # axs[1, 2].set(ylabel='Pixel Latitude')
    axs[1, 2].set(xlabel='Pixel Longitude')
    axs[1, 2].set(title='Band 7 (2105-2155 nm)')
    fig2.delaxes(axs[1, 3])
    fig2.suptitle(f'{instrument_name} : {datestr} ({jdate})', fontsize=16, fontweight='bold')
    fig2.tight_layout()
    plt.savefig(f'{outdir}bands_imaged/VMAX_{outname}.png')
    plt.close(fig2)

    # Do a 3-color png image
    band1_norm = normalize(array_3stack_float[:, :, 0])
    band4_norm = normalize(array_3stack_float[:, :, 1])
    band3_norm = normalize(array_3stack_float[:, :, 2])
    normalized_3stack = np.stack((band1_norm, band4_norm, band3_norm))
    normalized_3stack = np.swapaxes(normalized_3stack, axis1=0, axis2=1)
    normalized_3stack = np.swapaxes(normalized_3stack, axis1=1, axis2=2)

    fig3 = plt.figure(figsize=(8, 8))
    plt.imshow(normalized_3stack)
    plt.title(f'{instrument_name} : {datestr} ({jdate})', fontsize=16, fontweight='bold')
    plt.savefig(f'{outdir}png_image/{outname}_3band.png')
    plt.close(fig3)

    print(f'Processed {outname}...')

    return


# Main
input_directory = '/media/francesco/passport/fdl/modis_data/aqua/'
output_directory = '/media/francesco/passport/fdl/modis_output_aqua/'
if not os.path.exists(f'{output_directory}bands_imaged'):
    os.mkdir(f'{output_directory}bands_imaged')
if not os.path.exists(f'{output_directory}png_image'):
    os.mkdir(f'{output_directory}png_image')
if not os.path.exists(f'{output_directory}log_files'):
    os.mkdir(f'{output_directory}log_files')
if not os.path.exists(f'{output_directory}array_7bands'):
    os.mkdir(f'{output_directory}array_7bands')
if not os.path.exists(f'{output_directory}array_3bands'):
    os.mkdir(f'{output_directory}array_3bands')

filelist = glob.glob(f'{input_directory}*.hdf')

# for file in filelist:
#     process_hdf(file, output_directory)

num_cores = 3
Parallel(n_jobs=num_cores)(delayed(process_hdf)(file, output_directory) for file in filelist)
