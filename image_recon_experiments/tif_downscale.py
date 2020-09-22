"""
This code downscales an input geotiff into one of lower resolution

"""

import sys
sys.dont_write_bytecode = True

# Import packages
import rasterio
import rasterio.plot
import rasterio.features
import rasterio.warp
from rasterio.enums import Resampling
import numpy as np
import glob
import os
from matplotlib import pyplot as plt


# Functions
def plotting_resample(array):
    """
    Normalizes numpy arrays into scale 0.0 - 1.0
    NOTE: This is only for plotting
    """
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))


def our_resample(input_tif, scale_factor):
    '''
    Parameters
    ----------
    input_tif : [str] Input tif file
    scale_factor : [float] scaling factor for resampling

    Returns
    -------
    '''

    with rasterio.open(input_tif) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * scale_factor),
                int(dataset.width * scale_factor)
            ),
            resampling=Resampling.cubic
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    return data


def resample_wrapper(tif_band1, tif_band4, tif_band3, scale_factor, out_folder):
    '''

    Parameters
    ----------
    tif_band1 : [str] filename of tif band 1
    tif_band4 : [str] filename of tif band 4
    tif_band3 : [str] filename of tif band 3
    scale_factor : [float] scaling factor for resampling

    Returns combined numpy file
    -------

    '''

    # Run the resample for each band
    band1_resample = our_resample(tif_band1, scale_factor)
    band4_resample = our_resample(tif_band4, scale_factor)
    band3_resample = our_resample(tif_band3, scale_factor)

    # For plotting, use this normalization:
    # NOTE: This doesn't work yet. Fix later.
    # band1_resample = plotting_resample(our_resample(testfile_band1, scale_factor))
    # band4_resample = plotting_resample(our_resample(testfile_band4, scale_factor))
    # band3_resample = plotting_resample(our_resample(testfile_band3, scale_factor))

    resampled_bands = np.vstack([band1_resample, band4_resample, band3_resample])
    resampled_bands = np.swapaxes(resampled_bands, axis1=0, axis2=2)
    print(resampled_bands.shape)

    # Get the basename of the original files
    bname = os.path.basename(tif_band1).split('_')[0]
    filename = f'{bname}_sf{scale_factor}'
    np.save(f'{out_folder}{filename}.npy', resampled_bands, allow_pickle=True)

    print(f'Scaled file {filename}...')

    return


# Main
def main():
	data_folder = '/home/satyarth934/Projects/NASA_FDL_2020/featurizer/FDL_2020/delete/'
	output_folder = '/home/satyarth934/Projects/NASA_FDL_2020/featurizer/FDL_2020/delete_output/'

	band1_files = sorted(glob.glob(f'{data_folder}*_B01.TIF'))
	band4_files = sorted(glob.glob(f'{data_folder}*_B04.TIF'))
	band3_files = sorted(glob.glob(f'{data_folder}*_B03.TIF'))

	for num_file in np.arange(len(band1_files)):
	    resample_wrapper(band1_files[num_file], band4_files[num_file], band4_files[num_file], 0.25, output_folder)


if __name__ == '__main__':
	main()
