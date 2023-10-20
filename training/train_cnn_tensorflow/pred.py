import os
import json
from pathlib import Path
# import GPUtil
#DEVICE_ID_LIST = GPUtil.getFirstAvailable()
#os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID_LIST[0])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from osgeo import gdal
from models import att_r2_unet, get_coswin_model
import geopandas
import rasterio
from rasterio import windows
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
import tensorflow as tf
from tqdm import tqdm
from glob import glob


def normalize_meanstd(img, means=None, stds=None):
    return np.true_divide(np.subtract(img, means), stds)


def merge_prediction(pred_raster, save_win, model_pred, avg_filter):
    pred = pred_raster.read(window=save_win)
    pred, n_pred = pred[:-1], pred[-1]
    pred = np.moveaxis(pred, 0, -1)
    if len(model_pred.shape)==3:
        for i in range(model_pred.shape[-1]):
            model_pred[...,i] *= avg_filter
    else:
        model_pred *= avg_filter
        pred = np.squeeze(pred)
    pred += model_pred
    n_pred += avg_filter

    return pred, n_pred


def save_prediction(pred_raster, save_win, pred):
    for i in range(pred.shape[-1]):
        pred_raster.write(pred[:,:,i], window=save_win, indexes=i+1)


def get_new_index(idx, last_bool, done_bool, dim, load_size, pred_file, overlap=0.5):
    if last_bool:
        done_bool = True
    else:
        idx += int(load_size * (1.0-overlap))
        if idx+load_size >= pred_file.shape[dim]:
            idx = pred_file.shape[dim] - load_size
            last_bool = True
    return idx, last_bool, done_bool


def execute(annotations_index_path: str, imgs_dir: str, save_dir: str, hdf5_model_path: dict
            , model_name: str = 'attr2unet', img_res: float = 0.5, pred_res: float = 0.5, pred_size: int = 512
            , overlap: float = 0.5, n_labels: int = 1, batch_size: int = 32):
    # Define parameters
    load_size = int(pred_size * pred_res / img_res)

    # load model
    if model_name == 'attr2unet':
        model = att_r2_unet((pred_size,pred_size,4), n_labels=n_labels)
    model.load_weights(hdf5_model_path)
    model.trainable = False
    model.summary()

    # Create averaging filter
    avg_filter = np.zeros([pred_size, pred_size])
    avg_filter[int(pred_size/4):int(3*pred_size/4), int(pred_size/4):int(3*pred_size/4)] = 1
    avg_filter = cv2.GaussianBlur(avg_filter, (pred_size-1,pred_size-1), 0)
    avg_filter = avg_filter + 0.25
    avg_filter = avg_filter / avg_filter.max()
    avg_filter = avg_filter.astype(np.float32)

    # Visualize
    #plt.imshow(avg_filter)
    #plt.show()

    # Set provider metadata
    set_file = Path(Path.cwd() / "settings.json")
    if not set_file.is_file():
        raise ValueError("The JSON file path provided is not a file.")
    settings = type('Settings', (object,), json.loads(set_file.read_text(encoding='utf-8')))()

    # Read metadata
    ge_metadata = np.load(settings.ge_metadata_path)
    ge_means = ge_metadata['means']
    ge_stds = ge_metadata['stds']
    wv2_metadata = np.load(settings.wv2_metadata_path)
    wv2_means = wv2_metadata['means']
    wv2_stds = wv2_metadata['stds']
    wv3_metadata = np.load(settings.wv3_metadata_path)
    wv3_means = wv3_metadata['means']
    wv3_stds = wv3_metadata['stds']
    wv4_metadata = np.load(settings.wv4_metadata_path)
    wv4_means = wv4_metadata['means']
    wv4_stds = wv4_metadata['stds']

    # Read index geodataframe and zone names
    index_gdf = geopandas.read_file(annotations_index_path)

    # Loop in test paths
    # Get unique list of regions
    regions_list = []
    for name in index_gdf['nom_etendue']:
        name = name.replace('-', '_')
        name = name.split('_')[0]
        name = ''.join(letter for letter in name if not letter.isdigit())
        if name not in regions_list:
            regions_list.append(name)

    # Loop in regions
    for region in regions_list:

        # Get list of annotate zones in region
        zones_name = [name for name in index_gdf['nom_etendue'] if name[:len(region)] == region]
        for zone_name in tqdm(zones_name):
            if index_gdf['type_donnees'].loc[index_gdf['nom_etendue']==zone_name].values != 'benchmark':
                continue

            # Uniformization of zone names
            zone_name = zone_name.replace('-', '_')
            if '_QB02_' in zone_name or '_GE01_' in zone_name:  # Remove QuickBird and geoeye images
                continue
            print(zone_name)

            # Get bands and annotation paths
            zone, sat = zone_name.split('_')[:2]
            bands_path = []
            for channel in ['B', 'G', 'R', 'N']:
                bands_path += glob(imgs_dir + '**/' + zone + '_' + '*' + sat + '*' + channel + '.tif', recursive=True)
                bands_path += glob(imgs_dir + '**/' + zone + '-' + '*' + sat + '*' + channel + '.tif', recursive=True)
                bands_path += glob(imgs_dir + '**/' + zone + 'P00' + '*' + sat + '*' + channel + '.tif', recursive=True)

            # Read bands files
            bands_files = [rasterio.open(path) for path in bands_path]

            # Compute load size
            load_size = int(pred_size * pred_res / bands_files[0].transform[0])

            # init predictions file
            save_path = save_dir + zone_name + '_' + model_name + '_tfbaseline.tif'
            pred_kwds = bands_files[0].profile
            pred_kwds['count'] = n_labels+1
            pred_kwds['dtype'] = np.float32
            pred_kwds['nodata'] = 0.0
            pred_kwds['compress'] = 'lzw'
            pred_kwds['driver'] = 'GTiff'
            pred_file = rasterio.open(
                save_path,
                'w+',
                BIGTIFF=True,
                **pred_kwds
            )

            # loop in image
            idx_i, last_i, done_i = 0, False, False
            idx_j, last_j, done_j = 0, False, False
            pbar = tqdm(total=int(pred_file.shape[0] / pred_size / (1-overlap)))
            while not (done_j and done_i):

                # Create images batch
                img_batch = []
                img_windows = []
                while (len(img_batch) != batch_size):

                    # get default coordinates and mask window
                    win_w, win_s = rasterio.transform.xy(pred_file.transform, idx_i, idx_j)
                    win_e, win_n = rasterio.transform.xy(pred_file.transform, idx_i+load_size, idx_j+load_size)
                    pred_win = windows.from_bounds(win_w, win_n, win_e, win_s, pred_file.transform).round_offsets()

                    # read windowed image and skip if more than 0.5% border in image
                    sub_img = [file.read(window=pred_win) for file in bands_files]
                    sub_img = np.squeeze(np.asarray(sub_img))
                    nan_ratio = (sub_img==0).sum() / sub_img.shape[0] / sub_img.shape[1] / sub_img.shape[2]
                    if nan_ratio < 0.005 and not (np.isnan(sub_img)).any():

                        # dimension correction
                        sub_img = np.moveaxis(sub_img, 0, -1)

                        # resize if needed
                        if load_size != pred_size:
                            r_img = np.zeros((4, pred_size, pred_size))
                            for i in range(4):
                                r_img[i] = cv2.resize(sub_img[i], (pred_size, pred_size))
                            sub_img = r_img
                            del r_img

                        # normalize data
                        means, stds = None, None
                        if '_GE01_' in zone_name:
                            means = ge_means
                            stds = ge_stds
                        elif '_WV02_' in zone_name:
                            means = wv2_means
                            stds = wv2_stds
                        elif '_WV03_' in zone_name:
                            means = wv3_means
                            stds = wv3_stds
                        elif '_WV04_' in zone_name:
                            means = wv4_means
                            stds = wv4_stds
                        sub_img = normalize_meanstd(sub_img, means, stds)

                        # prepare dimensions for prediction
                        img_batch.append(sub_img)
                        img_windows.append(pred_win)

                    # get new indexes
                    idx_j, last_j, done_j = get_new_index(idx_j, last_j, done_j, 1, load_size, pred_file)
                    if done_j:
                        idx_i, last_i, done_i = get_new_index(idx_i, last_i, done_i, 0, load_size, pred_file)
                        pbar.update(1)
                        if not done_i:
                            idx_j, last_j, done_j = 0, False, False
                        else:
                            break

                # model prediction
                if len(img_windows) > 0:
                    img_batch = np.asarray(img_batch)
                    batch_pred = model(img_batch).numpy()

                    # write to mosaic
                    for ii in range(len(img_windows)):
                        # read prediction and position in raster
                        pred = batch_pred[ii]
                        img_win = img_windows[ii]

                        # resize prediction
                        if load_size != pred_size:
                            pred = cv2.resize(pred, (int(img_win.height), int(img_win.width)))

                        # merge prediction
                        pred, n_pred = merge_prediction(pred_file, img_win, pred, avg_filter)
                        pred = np.dstack([pred, n_pred])

                        # Save to prob rasters
                        save_prediction(pred_file, img_win, pred)

            # close files
            [band_file.close() for band_file in bands_files]
            pred_file.close()
            pbar.close()
