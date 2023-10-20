import argparse
import os
from osgeo import gdal
import rasterio
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def execute(mask_dir: str, save_dir: str, tile_size: int = 4096, overlap: int = 256) -> None:
    """
    Clip tiles from masks
    Parameters
    ----------
    mask_dir: Mask directory
    save_dir: Directory to save tiles
    tile_size: Size of tiles
    overlap: Overlap between tiles

    Returns
    -------
    """

    # Loop in masks directories
    for file in tqdm(os.listdir(mask_dir)):
        if file.split('.')[-1] != 'tif':
            continue

        # Load mask file
        with rasterio.open(mask_dir + file) as mask_file:
            kwds = mask_file.profile
            mask_transform = mask_file.transform
            mask = mask_file.read()[0]

        # Update new width and height in metadata
        kwds['height'] = tile_size
        kwds['width'] = tile_size

        # Clip tiles in mask
        idx_i = 0
        while idx_i+tile_size <= mask.shape[0]:
            idx_j = 0
            while idx_j+tile_size <= mask.shape[1]:

                # clip tile
                mask_tile = np.copy(mask[idx_i:idx_i+tile_size, idx_j:idx_j+tile_size])

                # update metadata
                west, south = rasterio.transform.xy(mask_file.transform, idx_i, idx_j, offset='ll')
                east, north = rasterio.transform.xy(mask_file.transform, idx_i+tile_size, idx_j+tile_size, offset='ll')
                tile_transform = rasterio.transform.from_bounds(west, north, east, south, tile_size, tile_size)
                kwds['transform'] = tile_transform

                # save tile
                save_path = save_dir + file[:-9] + '_' + str(idx_i) + '_' + str(idx_j) + '_mask.tif'
                with rasterio.open(save_path, 'w', **kwds) as save_file:
                    save_file.write_band(1, mask_tile)
                # plt.imshow(mask_tile), plt.show()   # For visialization purpose

                # update indexes
                idx_j += tile_size - overlap
            idx_i += tile_size - overlap
