import os
import numpy as np
from osgeo import gdal
import rasterio
from tqdm import tqdm
from glob import glob
from matplotlib import pyplot as plt


def execute(data_dir: str, save_dir: str, file_name: str, satellites: list[str], bands: list[str]):
    """
    Get satellites statistics from input repository
    """

    # initialize statistics
    mean_list = [[] for i in range(4)]
    var_list = [[] for i in range(4)]
    max_list = [[] for i in range(4)]

    print(' --- Computing means and maxs --- ')
    # Get maximum and mean stats
    for sat in satellites:
        if not os.path.exists(save_dir+sat+file_name):
            print(' ---', sat, '--- ')

            # Get stats for every bands
            for i, band in enumerate(bands):

                # get images paths
                paths = glob(data_dir+sat+'/*'+band+'.tif')
                for path in tqdm(paths):

                    # read image array
                    with rasterio.open(path) as img_file:
                        img = img_file.read()

                    # Remove border
                    img = np.float32(img)
                    img[img == 0] = np.nan

                    # append statistics
                    mean_list[i].append(np.nanmean(img))
                    max_list[i].append(np.nanmax(img))

            # Calculate datasets statistics
            means, maxs = [], []
            for i in range(len(bands)):
                means.append(np.mean(mean_list[i]))
                maxs.append(np.max(max_list[i]))

            # save meta data
            np.savez(
                save_dir + sat+file_name,
                means=means,
                maxs=maxs,
                )

    print(' --- Computing stds --- ')
    # Get std stats
    for sat in satellites:
        print(' ---', sat, '--- ')

        # Read maximums and means
        metadata = np.load(save_dir+sat+file_name)
        maxs = metadata['maxs']
        means = metadata['means']

        # Get stats for every bands
        for i, band in enumerate(bands):

            # get images paths
            paths = glob(data_dir + sat+'/*' + band + '.tif')
            for path in tqdm(paths):

                # read image array
                with rasterio.open(path) as img_file:
                    img = img_file.read()

                # Remove border
                img = np.float32(img)
                img[img == 0] = np.nan

                # append statistics
                var_list[i].append(np.nanmean((img - means[i]) ** 2))

        # Calculate datasets statistics
        stds = []
        for i in range(len(bands)):
            stds.append(np.sqrt(np.nanmean(var_list[i])))

        print('means :', means)
        print('stds :', stds)
        print('maxs :', maxs)

        # save meta data
        np.savez(
            save_dir + sat + file_name,
            means=means,
            stds=stds,
            maxs=maxs,
            )
