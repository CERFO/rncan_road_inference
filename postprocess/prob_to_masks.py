import os
from osgeo import gdal
import rasterio
from rasterio import features
import geopandas
import numpy as np
from utils import clip_to_min_extent
from glob import glob
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt


def execute(annotations_index_path: str, gt_dir: str, preds_dir: str, save_dir: str, model_name: str
            , nodata_val: int=255, prob_threshold: float=0.5):
    """
    Create masks for comparison
    """

    if not os.path.exists(annotations_index_path):
        raise ValueError('annotations_index_path does not exist.')
    if not os.path.exists(gt_dir):
        raise ValueError('gt_dir does not exist.')
    if not os.path.exists(preds_dir):
        raise ValueError('preds_dir does not exist.')
    if not os.path.exists((Path(save_dir) / model_name).as_posix()):
        os.makedirs((Path(save_dir) / model_name))
    if not os.path.exists((Path(save_dir) / 'GT').as_posix()):
        os.makedirs((Path(save_dir) / 'GT'))

    # Loop in predictions folder
    preds_dir += model_name + '/'
    for file in tqdm(os.listdir(preds_dir)):
        if file[-4:] != '.tif':
            continue

        # Get image name and ground truth path
        img_name = '_'.join(file.split('_')[:3])
        gt_path = save_dir + 'GT/' + img_name + '_mask.tif'
        save_path = save_dir + model_name + '/' + img_name + '_mask.tif'

        # Read prediction
        pred_file = rasterio.open(preds_dir + file)
        pred = pred_file.read()
        kwds = pred_file.profile

        # Prediction normalization and thresholding
        pred = np.true_divide(pred[0], pred[-1], where=pred[-1]!=0)
        pred = np.where(pred>=prob_threshold, 1, 0).astype(np.uint8)

        # Clip to minimum annotated zone for comparison
        pred, kwds = clip_to_min_extent(annotations_index_path, pred, file, pred_file.crs, kwds, nodata_val)

        # Save to raster
        with rasterio.open(save_path, 'w', **kwds) as save_file:
            save_file.write_band(1, pred)

        # Save ground truth if not done
        if not os.path.exists(gt_path):

            # Get annotation geodataframe
            annotation_path = glob(gt_dir + '*'.join(img_name.split('_')) + '*.gpkg')[0]
            annotation_gdf = geopandas.read_file(annotation_path).to_crs(kwds['crs'])
            shapes = ((geom.buffer(float(buffer_size) if not np.isnan(float(buffer_size)) else 3.0),1) for geom, buffer_size in zip(annotation_gdf.geometry, annotation_gdf['largeur_buffer']))
            mask = features.rasterize(shapes=shapes, fill=0, out_shape=pred.shape, dtype=np.uint8, transform=kwds['transform'])

            # Crop raster
            mask[pred==nodata_val] = nodata_val

            # Save ground truth
            with rasterio.open(gt_path, 'w', **kwds) as save_file:
                save_file.write_band(1, mask)
