import os
from osgeo import gdal
import rasterio
from rasterio import features, windows
import geopandas
import numpy as np
from glob import glob
from tqdm import tqdm
import cv2
import tifffile
import random
from matplotlib import pyplot as plt


def execute(annotations_index_path: str, annotations_dir: str, img_dir: str, save_dir: str
            , wanted_size: int = 512, wanted_res: float = 0.5, overlap_train: float = 0.3, overlap_val: float = 0.3
            , val_prop: float = 0.20):
    """
    Create dataset from annotations
    """

    # Create dataset folders
    save_dir += 'size' + str(wanted_size) + '_res' + str(wanted_res) + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_dir + 'train/'):
        os.mkdir(save_dir + 'train/')
    if not os.path.exists(save_dir + 'train/true_data/'):
        os.mkdir(save_dir + 'train/true_data/')
    if not os.path.exists(save_dir + 'train/false_data/'):
        os.mkdir(save_dir + 'train/false_data/')
    if not os.path.exists(save_dir + 'val/'):
        os.mkdir(save_dir + 'val/')
    if not os.path.exists(save_dir + 'val/true_data/'):
        os.mkdir(save_dir + 'val/true_data/')
    if not os.path.exists(save_dir + 'val/false_data/'):
        os.mkdir(save_dir + 'val/false_data/')

    # Read index geodataframe
    index_gdf = geopandas.read_file(annotations_index_path)

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
        print('', region)

        # Get list of annotate zones in region
        zones_name = [name for name in index_gdf['nom_etendue'] if name[:len(region)] == region]

        #random.shuffle(zones_name)
        for zone_name in tqdm(zones_name):
            if index_gdf['type_donnees'].loc[index_gdf['nom_etendue']==zone_name].values == 'benchmark':
                continue

            # Uniformization of zone names
            zone_name = zone_name.replace('-', '_')

            # Skip some satellites if wanted
            if '_QB02_' in zone_name:
                continue

            # Decide if validation or training zone
            n_imgs_train = len(glob(save_dir + 'train/**/*.tif'))
            n_imgs_val = len(glob(save_dir + 'val/**/*.tif'))
            n_imgs_total = n_imgs_train + n_imgs_val
            if n_imgs_total == 0:
                dataset_type = 'val/'
            else:
                dataset_type = 'val/' if n_imgs_val/n_imgs_total < val_prop else 'train/'
            overlap = overlap_train if dataset_type == 'train/' else overlap_val

            # Get bands and annotation paths
            zone, sat = zone_name.split('_')[:2]
            bands_path = []
            for channel in ['B', 'G', 'R', 'N']:
                bands_path += glob(img_dir + '**/' + zone + '_' + '*' + sat + '*' + channel + '.tif', recursive=True)
                bands_path += glob(img_dir + '**/' + zone + '-' + '*' + sat + '*' + channel + '.tif', recursive=True)
                bands_path += glob(img_dir + '**/' + zone + 'P00' + '*' + sat + '*' + channel + '.tif', recursive=True)
            annotation_path = glob(annotations_dir + zone + '*' + sat + '*.gpkg')[0]

            # Read bands files
            bands_files = [rasterio.open(path) for path in bands_path]

            # Rasterize annotated zone
            zone_gdf = index_gdf.to_crs(bands_files[0].crs)
            shapes = ((geom,1) for geom in zone_gdf.geometry)
            zone = features.rasterize(shapes=shapes, fill=0, out_shape=bands_files[0].shape, dtype=np.uint8, transform=bands_files[0].transform)

            # Read annotation and rasterize
            annotation_gdf = geopandas.read_file(annotation_path)
            shapes = ((geom.buffer(float(buffer_size) if not np.isnan(float(buffer_size)) else 3.0),1) for geom, buffer_size in zip(annotation_gdf.geometry, annotation_gdf['largeur_buffer']))
            mask = features.rasterize(shapes=shapes, fill=0, out_shape=bands_files[0].shape, dtype=np.uint8, transform=bands_files[0].transform)

            # Compute load size
            load_size = int(wanted_size * wanted_res / bands_files[0].transform[0])

            # Crop patches from image
            idx_i = 0
            while idx_i + load_size < zone.shape[0]:
                idx_j = 0
                while idx_j + load_size < zone.shape[1]:

                    # check if in annotated zone
                    sub_zone = np.copy(zone[idx_i:idx_i+load_size, idx_j:idx_j+load_size])
                    if (sub_zone == 1).all():

                        # read windowed image and skip if more than 0.5% border in image
                        sub_img = [file.read(window=windows.Window(idx_j, idx_i, load_size, load_size)) for file in bands_files]
                        sub_img = np.squeeze(np.asarray(sub_img))
                        nan_ratio = (sub_img==0).sum() / sub_img.shape[0] / sub_img.shape[1] / sub_img.shape[2]
                        sub_mask = np.copy(mask[idx_i:idx_i+load_size, idx_j:idx_j+load_size])
                        if nan_ratio < 0.005 and not (np.isnan(sub_img)).any() and sub_img.shape[1:] == sub_mask.shape:

                            # resize if needed
                            if load_size != wanted_size:
                                r_img = np.zeros((4, wanted_size, wanted_size))
                                for i in range(4):
                                    r_img[i] = cv2.resize(sub_img[i], (wanted_size, wanted_size))
                                sub_img = r_img
                                del r_img
                                sub_mask = cv2.resize(sub_mask, (wanted_size, wanted_size), interpolation=cv2.INTER_NEAREST)

                            # decide save folder
                            save_folder = 'true_data/' if sub_mask.sum() / sub_mask.shape[0] / sub_mask.shape[1] > 0.0015 else 'false_data/'

                            # save data
                            if save_folder == 'true_data/' or np.random.random() < 0.1:
                                data = np.concatenate([sub_img, np.expand_dims(sub_mask, 0)], axis=0)
                                data_name = zone_name + '_' + str(idx_i) + '_' + str(idx_j) + '.tif'
                                tifffile.imwrite(save_dir + dataset_type + save_folder + data_name, data)

                            # Visualization for debug purpose :
                            # if sub_mask.sum() > 0:
                                # print(data_name)
                                # print(nan_ratio)
                                # plt.subplot(121), plt.imshow(data[0])
                                # plt.subplot(122), plt.imshow(data[0]), plt.imshow(data[-1], alpha=0.4)
                                # plt.show()
                            # End of debug

                    idx_j += int(wanted_size * (1-overlap))
                idx_i += int(wanted_size * (1-overlap))
