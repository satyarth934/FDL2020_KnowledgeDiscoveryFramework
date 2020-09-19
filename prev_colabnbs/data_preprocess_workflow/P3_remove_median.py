"""
This code removes the median and mean (from P2) from the normalized tiles (from P1)

"""
# Import packages
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from joblib import Parallel, delayed


# Setup function
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min = np.nanmin(array)
    array_max = np.nanmax(array)
    return (array - array_min) / (array_max - array_min)


def remove_median(input_array, median_array):
    """
    Function to remove the median from an input array

    Parameters
    ----------
    input_array : [np array] Input array which needs the median removed
    median_array : [np array] Median array

    Returns
    -------

    """
    output_array = np.subtract(input_array, median_array)

    return output_array


def remove_mean_stdev(input_array, mean_array, stdev_array):
    """
    Subtracts the mean and divides by the standard deviation

    Parameters
    ----------
    input_array : [np array] Input array to remove the mean/stdev
    mean_array : [np array] Mean array
    stdev_array : [np array] Standard deviation array

    Returns
    -------
    """

    # First, subtract the mean
    output_array = np.subtract(input_array, mean_array)

    # Then do a pixel-wise division
    output_array = np.divide(output_array, stdev_array)

    return output_array


def remove_median_mean(infile, input_pixel_size, outdir):

    """

    Parameters
    ----------
    infile : [str] Input file to be processed
    input_pixel_size : [int] Image length in pixels for the file

    Returns
    -------

    """
    # Get the filename in a string
    bname = os.path.basename(infile)
    bname_condensed = bname.split('_')[1]

    # Load the input array
    data = np.load(infile)

    # Load the file corresponding to the median, mean, and stdev for this pixel size
    band1_median = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_1_median.npy')[0])
    band4_median = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_4_median.npy')[0])
    band3_median = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_3_median.npy')[0])

    band1_mean = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_1_mean.npy')[0])
    band4_mean = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_4_mean.npy')[0])
    band3_mean = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_3_mean.npy')[0])

    band1_stdev = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_1_stdev.npy')[0])
    band4_stdev = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_4_stdev.npy')[0])
    band3_stdev = np.load(glob.glob(f'{stat_directory}{input_pixel_size}_band_3_stdev.npy')[0])

    # Remove the median from each band
    band1_median_removed = remove_median(data[:, :, 0], band1_median)
    band4_median_removed = remove_median(data[:, :, 1], band4_median)
    band3_median_removed = remove_median(data[:, :, 2], band3_median)

    # Remove the mean/stdev from each band
    band1_mean_removed = remove_mean_stdev(data[:, :, 0], band1_mean, band1_stdev)
    band4_mean_removed = remove_mean_stdev(data[:, :, 1], band4_mean, band4_stdev)
    band3_mean_removed = remove_mean_stdev(data[:, :, 2], band3_mean, band3_stdev)

    # Stack the new results
    median_removed_3stack = np.stack((band1_median_removed, band4_median_removed, band3_median_removed))
    median_removed_3stack = np.swapaxes(median_removed_3stack, axis1=0, axis2=1)
    median_removed_3stack = np.swapaxes(median_removed_3stack, axis1=1, axis2=2)

    mean_removed_3stack = np.stack((band1_mean_removed, band4_mean_removed, band3_mean_removed))
    mean_removed_3stack = np.swapaxes(mean_removed_3stack, axis1=0, axis2=1)
    mean_removed_3stack = np.swapaxes(mean_removed_3stack, axis1=1, axis2=2)

    # Save the results
    np.save(f'{outdir}{input_pixel_size}/median_removed/{input_pixel_size}_{bname_condensed}_nomedian.npy',
            median_removed_3stack, allow_pickle=True)
    np.save(f'{outdir}{input_pixel_size}/mean_stdev_removed/{input_pixel_size}_{bname_condensed}_nomean.npy',
            mean_removed_3stack, allow_pickle=True)

    # Plot results
    # Full plot of each band
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))
    axs[0, 0].imshow(data[:, :, 0], cmap=cm.jet)
    axs[0, 0].set(title='Band1 Original')
    axs[0, 1].imshow(median_removed_3stack[:, :, 0], cmap=cm.jet)
    axs[0, 1].set(title='Band1 No Median')
    axs[0, 2].imshow(mean_removed_3stack[:, :, 0], cmap=cm.jet)
    axs[0, 2].set(title='Band1 No Mean/Stdev')

    axs[1, 0].imshow(data[:, :, 1], cmap=cm.jet)
    axs[1, 0].set(title='Band4 Original')
    axs[1, 1].imshow(median_removed_3stack[:, :, 1], cmap=cm.jet)
    axs[1, 1].set(title='Band4 No Median')
    axs[1, 2].imshow(mean_removed_3stack[:, :, 1], cmap=cm.jet)
    axs[1, 2].set(title='Band4 No Mean/Stdev')

    axs[2, 0].imshow(data[:, :, 2], cmap=cm.jet)
    axs[2, 0].set(title='Band3 Original')
    axs[2, 1].imshow(median_removed_3stack[:, :, 2], cmap=cm.jet)
    axs[2, 1].set(title='Band3 No Median')
    axs[2, 2].imshow(mean_removed_3stack[:, :, 2], cmap=cm.jet)
    axs[2, 2].set(title='Band3 No Mean/Stdev')

    fig.tight_layout()
    plt.savefig(f'{outdir}{input_pixel_size}/plots/FULLPLOT_{input_pixel_size}_{bname_condensed}.png')
    plt.close(fig)

    # Plot the results not-normalized
    # fig2, axs = plt.subplots(1, 3, figsize=(10, 5))
    # axs[0].imshow(data, cmap=cm.jet)
    # axs[0].set(title='Original')
    #
    # axs[1].imshow(median_removed_3stack, cmap=cm.jet)
    # axs[1].set(title='Median Removed NoNorm')
    #
    # axs[2].imshow(mean_removed_3stack, cmap=cm.jet)
    # axs[2].set(title='Mean Removed NoNorm')
    #
    # fig2.tight_layout()
    # plt.savefig(f'{outdir}{input_pixel_size}/plots/NONORM_{input_pixel_size}_{bname_condensed}.png')
    # plt.close(fig2)

    # Condensed plot of values
    # For plotting purposes, we will normalize the data again
    band1_nomedian_norm = normalize(median_removed_3stack[:, :, 0])
    band4_nomedian_norm = normalize(median_removed_3stack[:, :, 1])
    band3_nomedian_norm = normalize(median_removed_3stack[:, :, 2])
    nomedian_normalized_3stack = np.stack((band1_nomedian_norm, band4_nomedian_norm, band3_nomedian_norm))
    nomedian_normalized_3stack = np.swapaxes(nomedian_normalized_3stack, axis1=0, axis2=1)
    nomedian_normalized_3stack = np.swapaxes(nomedian_normalized_3stack, axis1=1, axis2=2)

    band1_nomean_norm = normalize(mean_removed_3stack[:, :, 0])
    band4_nomean_norm = normalize(mean_removed_3stack[:, :, 1])
    band3_nomean_norm = normalize(mean_removed_3stack[:, :, 2])
    nomean_normalized_3stack = np.stack((band1_nomean_norm, band4_nomean_norm, band3_nomean_norm))
    nomean_normalized_3stack = np.swapaxes(nomean_normalized_3stack, axis1=0, axis2=1)
    nomean_normalized_3stack = np.swapaxes(nomean_normalized_3stack, axis1=1, axis2=2)

    fig3, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(data, cmap=cm.jet)
    axs[0].set(title='Original')

    axs[1].imshow(nomedian_normalized_3stack, cmap=cm.jet)
    axs[1].set(title='Median Removed')

    axs[2].imshow(nomean_normalized_3stack, cmap=cm.jet)
    axs[2].set(title='Mean Removed')

    fig3.tight_layout()
    plt.savefig(f'{outdir}{input_pixel_size}/plots/COLORPLOT_{input_pixel_size}_{bname_condensed}.png')
    plt.close(fig3)

    print(f'Finished processing {input_pixel_size} {bname_condensed}...')

    return


def computation_wrapper(input_dir, pixel_sizes, outdir):
    """

    Parameters
    ----------
    input_dir : [str] Input directory of the normalized data
    pixel_sizes : [vector] Pixel length/widths used

    Returns
    -------

    """

    # Cycle through the pixel sizes
    for pixel_size in pixel_sizes:

        # Get all the normalized files for the length/width described by pixel size
        filelist = glob.glob(f'{input_dir}{pixel_size}/*.npy')

        # Non-parallel version
        # for file in filelist:
        #     remove_median_mean(file, pixel_size, outdir)

        # Parallel version
        num_cores = 4
        Parallel(n_jobs=num_cores)(delayed(remove_median_mean)(file, pixel_size, outdir)
                                   for file in filelist)
    return


# Main
input_directory = '/media/francesco/passport/fdl/modis_output_terra/array_3bands_normalized/'
output_directory = '/media/francesco/passport/fdl/modis_output_terra/array_3bands_adapted/'
stat_directory = '/media/francesco/passport/fdl/modis_output_terra/array_med_mean_stdev/'
pixel_bands = [224, 448, 672]

# Create output directories if they don't exist for the two data products and pixel bands
if not os.path.exists(output_directory):
    os.mkdir(output_directory)
for pixel_band in pixel_bands:
    if not os.path.exists(f'{output_directory}{pixel_band}'):
        os.mkdir(f'{output_directory}{pixel_band}')
    if not os.path.exists(f'{output_directory}{pixel_band}/median_removed'):
        os.mkdir(f'{output_directory}{pixel_band}/median_removed')
    if not os.path.exists(f'{output_directory}{pixel_band}/mean_stdev_removed'):
        os.mkdir(f'{output_directory}{pixel_band}/mean_stdev_removed')
    if not os.path.exists(f'{output_directory}{pixel_band}/plots'):
        os.mkdir(f'{output_directory}{pixel_band}/plots')

# Compute the two products
computation_wrapper(input_directory, pixel_bands, output_directory)
