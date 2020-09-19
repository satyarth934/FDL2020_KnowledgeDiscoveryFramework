"""
This code computes the median, mean, and standard deviation for each pixel for a tile through time

FC 7/25/2020

"""
# Import packages
import os
import glob
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


# Functions
def load_band(array_file, band_string):
    """
    Only returns the band of interest

    Parameters
    ----------
    array_file : [str] Filepath of array containing numerous bands
    band_string : [str] String corresponding to the band we are intereted in

    Returns The 2D band of interest
    -------

    """
    array = np.load(array_file)

    if band_string == 'band_1':
        band_values = array[:, :, 0]
    if band_string == 'band_4':
        band_values = array[:, :, 1]
    if band_string == 'band_3':
        band_values = array[:, :, 2]

    return band_values


def compute_stats(indir, band_string, band_size, savedir):
    """
    Function to find the median, mean, and stdev for the files in the given directory for a specific band.
    Note that we will be operating on the 3-band data, which is composed of bands 1, 4, and 3 (RGB).
    Therefore, we have the following index designations:
    band1 = array[:, :, 0]
    band4 = array[:, :, 1]
    band3 = array[:, :, 2]

    Parameters
    ----------
    indir : [str] Input directory containing the normalized 3-band numpy arrays created in P1.
    band_string : [str] String corresponding to R, G, or B channels

    Returns
    -------

    """
    # Get a list of all of the files
    flist = glob.glob(f'{indir}/*.npy')

    # We don't want to keep all 3 channels in RAM
    # Therefore, we are going to use a different function to only return one channel
    for file_ind in np.arange(len(flist)):
        band = load_band(flist[file_ind], band_string)

        # Before we start, we need to reshape the band to include the third dimension
        # If not, it will not stack properly
        band = band.reshape(1, np.shape(band)[0], np.shape(band)[0])

        if file_ind == 0:
            channel_stack = band
        else:
            channel_stack = np.vstack((channel_stack, band))

    # Reshuffle the array so that it's arranged such that (lat, lon, channel)
    channel_stack = np.swapaxes(channel_stack, axis1=0, axis2=1)
    channel_stack = np.swapaxes(channel_stack, axis1=1, axis2=2)

    # For each pixel, find the median, mean, and stdev
    slice_median = np.zeros((np.shape(channel_stack)[0], np.shape(channel_stack)[1]))
    slice_mean = np.zeros((np.shape(channel_stack)[0], np.shape(channel_stack)[1]))
    slice_stdev = np.zeros((np.shape(channel_stack)[0], np.shape(channel_stack)[1]))

    for i in np.arange(np.shape(channel_stack)[0]):
        for j in np.arange(np.shape(channel_stack)[1]):
            slice_median[i, j] = np.nanmedian(channel_stack[i, j, :])
            slice_mean[i, j] = np.nanmean(channel_stack[i, j, :])
            slice_stdev[i, j] = np.nanstd(channel_stack[i, j, :])

    # Save results as numpy files
    np.save(f'{savedir}{band_size}_{band_string}_median', slice_median, allow_pickle=True)
    np.save(f'{savedir}{band_size}_{band_string}_mean', slice_mean, allow_pickle=True)
    np.save(f'{savedir}{band_size}_{band_string}_stdev', slice_stdev, allow_pickle=True)

    return slice_median, slice_mean, slice_stdev


def computation_wrapper(indir, band_sizes, outdir):
    """
    This is the wrapper function that will direct the processing to do for the files

    Parameters
    ----------
    indir : [str] Input directory containing all of the files to process
    band_sizes : [vector] Pixel-wide lengths (integers) to process

    Returns
    -------

    """
    # Cycle through the input bands
    for band_size in band_sizes:
        band1_median, band1_mean, band1_stdev = compute_stats(f'{indir}{band_size}', 'band_1', band_size, outdir)
        band4_median, band4_mean, band4_stdev = compute_stats(f'{indir}{band_size}', 'band_4', band_size, outdir)
        band3_median, band3_mean, band3_stdev = compute_stats(f'{indir}{band_size}', 'band_3', band_size, outdir)

        # Plot the results
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))
        axs[0, 0].imshow(band1_median[:, :], cmap=cm.jet)
        axs[0, 0].set(title='Band1 Median')
        axs[0, 1].imshow(band1_mean[:, :], cmap=cm.jet)
        axs[0, 1].set(title='Band1 Mean')
        axs[0, 2].imshow(band1_stdev[:, :], cmap=cm.jet)
        axs[0, 2].set(title='Band1 Stdev')

        axs[1, 0].imshow(band4_median[:, :], cmap=cm.jet)
        axs[1, 0].set(title='Band4 Median')
        axs[1, 1].imshow(band4_mean[:, :], cmap=cm.jet)
        axs[1, 1].set(title='Band4 Mean')
        axs[1, 2].imshow(band4_stdev[:, :], cmap=cm.jet)
        axs[1, 2].set(title='Band4 Stdev')

        axs[2, 0].imshow(band3_median[:, :], cmap=cm.jet)
        axs[2, 0].set(title='Band3 Median')
        axs[2, 1].imshow(band3_mean[:, :], cmap=cm.jet)
        axs[2, 1].set(title='Band3 Mean')
        axs[2, 2].imshow(band3_stdev[:, :], cmap=cm.jet)
        axs[2, 2].set(title='Band3 Stdev')

        fig.tight_layout()
        plt.savefig(f'{outdir}{band_size}_plot_stats.png')
        plt.close(fig)

        print(f'Finished processing band {band_size}...')

    return


# Main
input_directory = '/media/francesco/passport/fdl/modis_output_terra/array_3bands_normalized/'
output_directory = '/media/francesco/passport/fdl/modis_output_terra/array_med_mean_stdev/'
pixel_bands = [224, 448, 672]

# Create output directories if they don't exist for the two data products and pixel bands
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
# for pixel_band in pixel_bands:
#     if not os.path.exists(f'{output_directory}{pixel_band}'):
#         os.mkdir(f'{output_directory}{pixel_band}')
#     if not os.path.exists(f'{output_directory}{pixel_band}/median_removed'):
#         os.mkdir(f'{output_directory}{pixel_band}/median_removed')
#     if not os.path.exists(f'{output_directory}{pixel_band}/mean_stdev_removed'):
#         os.mkdir(f'{output_directory}{pixel_band}/mean_stdev_removed')

computation_wrapper(input_directory, pixel_bands, output_directory)
