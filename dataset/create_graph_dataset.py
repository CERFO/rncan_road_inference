import os
from osgeo import gdal
import rasterio
from rasterio import windows
import networkx as nx
import numpy as np
import scipy
import cv2
import tifffile
from graph_utils import create_graph_midpoints, decode_graph_encoding
from plot_road import plot_graph
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm


def execute(img_dir: str, gt_dir: str, save_dir: str, bands_order: list[str] | None
            , img_size: int = 512, overlap: float = 0.33, max_degree: int = 6, r: int = 1
            , linestring_delta_meters: int = 20):
    """
    Create graph dataset from annotations
    """

    # Get training, validation and test zones
    train_val_zones = np.load(os.getcwd() + '/train_val_zones.npz')
    train_zones = list(train_val_zones['train_zones'])
    val_zones = list(train_val_zones['val_zones'])

    # Loop in ground truth directory
    for file in tqdm(os.listdir(gt_dir)):

        if not '_graph.gpickle' in file:
            continue

        # Check if in test zones
        zone_name = '_'.join(file.split('_')[:3])
        if zone_name not in train_zones and zone_name not in val_zones:
            continue

        # Remove QuickBird and Geoeye images
        # if '_QB02_' in zone_name or '_GE01_' in zone_name:
        if not '_WV02_' in zone_name:
            continue

        # Read mask
        mask_path = gt_dir + file.replace('_graph.gpickle', '_mask.tif')
        mask_file = rasterio.open(mask_path)
        mask = mask_file.read()[0]

        # Get satellite bands files
        zone, sat = zone_name.split('_')[:2]
        bands_path = []
        for channel in bands_order:
            bands_path += glob(img_dir + '**/' + zone + '_' + '*' + sat + '*' + channel + '.tif', recursive=True)
            bands_path += glob(img_dir + '**/' + zone + '-' + '*' + sat + '*' + channel + '.tif', recursive=True)
            bands_path += glob(img_dir + '**/' + zone + 'P00' + '*' + sat + '*' + channel + '.tif', recursive=True)
        bands_files = [rasterio.open(band_path) for band_path in bands_path]
        vector_norm = 1/np.abs(bands_files[0].transform[0]) * linestring_delta_meters

        # Read satellite image in mask extent
        win_w, win_s = rasterio.transform.xy(mask_file.transform, 0, 0)
        win_e, win_n = rasterio.transform.xy(mask_file.transform, mask_file.shape[0], mask_file.shape[1])
        img_win = windows.from_bounds(win_w, win_n, win_e, win_s, bands_files[0].transform).round_offsets().round_shape()
        img = [band_file.read(window=img_win)[0] for band_file in bands_files]
        img = np.moveaxis(np.asarray(img).squeeze(), 0, -1)

        # Read ground truth graph
        g = nx.read_gpickle(gt_dir + file)
        n_nodes = len(g.nodes())

        # Add midpoints in graph to densify nodes
        g = create_graph_midpoints(g, mask_file, linestring_delta=linestring_delta_meters)

        # Initialize probabilities and vector arrays
        tiles_prob = np.zeros((mask.shape[0], mask.shape[1], 2*(max_degree+1)), dtype=np.uint8)
        tiles_vector = np.zeros((mask.shape[0], mask.shape[1], 2*(max_degree)))
        tiles_prob[...,1::2] = 1

        # Graph-tensor encoding
        if n_nodes > 0:
            for x_g, y_g in zip(g.nodes(data='x_pix'), g.nodes('y_pix')):
                node_id = int(x_g[0])
                x_g, y_g = int(x_g[1]), int(y_g[1])
                if x_g < 8 or y_g < 8 or x_g > mask.shape[1]-8 or y_g > mask.shape[0]-8:
                    continue
                # tiles_prob encoding for current node
                tiles_prob[y_g,x_g,0] = 1
                tiles_prob[y_g,x_g,1] = 0
                for x in range(x_g-r, x_g+r+1):
                    for y in range(y_g-r, y_g+r+1):
                        tiles_prob[y,x,0] = 1
                        tiles_prob[y,x,1] = 0
                # encoding for node's neighbors
                for neighbor_id in g.neighbors(node_id):
                    x_n, y_n = int(g.nodes[neighbor_id]['x_pix']), int(g.nodes[neighbor_id]['y_pix'])
                    if x_n < 8 or y_n < 8 or x_n > mask.shape[1]-8 or y_n > mask.shape[0]-8:
                        continue
                    # get angle and index of encoding
                    d = np.arctan2(y_n - y_g, x_n - x_g) + np.pi
                    j = int(d/(np.pi/3.0)) % max_degree
                    for x in range(x_g-r, x_g+r+1):
                        for y in range(y_g-r, y_g+r+1):
                            tiles_prob[y,x,2+2*j] = 1
                            tiles_prob[y,x,2+2*j+1] = 0
                            tiles_vector[y,x,2*j] = (y_n - y_g)/vector_norm
                            tiles_vector[y,x,2*j+1] = (x_n - x_g)/vector_norm

        # Merge prob and vector arrays
        imagegraph = []
        imagegraph.append(tiles_prob[..., 0:2])
        for i in range(max_degree):
            imagegraph.append(tiles_prob[..., 2+i*2:2+i*2+2])
            imagegraph.append(tiles_vector[..., i*2:i*2+2])
        del tiles_prob, tiles_vector
        imagegraph = np.concatenate(imagegraph, axis=2, dtype=np.float32).astype(np.float32)

        # Compact encoding of GTE
        compact_gte = np.zeros([imagegraph.shape[0], imagegraph.shape[1], 19])
        compact_gte[...,0] = imagegraph[...,0]
        for i in range(max_degree):
            compact_gte[..., 3*i + 1] = imagegraph[..., 4*i + 2]
            compact_gte[..., 3*i + 2] = imagegraph[..., 4*i + 4]
            compact_gte[..., 3*i + 3] = imagegraph[..., 4*i + 5]
        if compact_gte.min() < -1 or compact_gte.max() > 1:
            compact_gte[compact_gte<-1] = -1
            compact_gte[compact_gte>1] = 1
        imagegraph = compact_gte
        del compact_gte

        # Save sub images in dataset
        idx_i = 0
        while idx_i+img_size <= mask.shape[0]:
            idx_j = 0
            while idx_j+img_size <= mask.shape[1]:

                # get arrays
                sub_mask = np.copy(mask[idx_i:idx_i+img_size, idx_j:idx_j+img_size])
                if not (sub_mask == 255).any():
                    sub_img = np.copy(img[idx_i:idx_i+img_size, idx_j:idx_j+img_size])
                    sub_graph = np.copy(imagegraph[idx_i:idx_i+img_size, idx_j:idx_j+img_size])
                    sub_mask = np.expand_dims(sub_mask, -1)

                    # save to tifffile
                    dataset_type = 'train/' if zone_name in train_zones else 'val/'
                    save_folder = 'false_data/' if sub_mask.sum() < 100 else 'true_data/'
                    save_name = file.replace('_graph.gpickle', '') + '_' + str(idx_i) + '_' + str(idx_j) + '.tif'
                    data = np.dstack([sub_img, sub_graph])
                    if (save_folder == 'false_data/' and np.random.random() < 0.1) or save_folder == 'true_data/':
                        tifffile.imwrite(save_dir + dataset_type + save_folder + save_name, data)

                    # region DEBUG - Decode graph encoding
                    # if data[...,4].sum() > 0:
                        # plt.subplot(221), plt.imshow(data[...,0])
                        # plt.subplot(222), plt.imshow(data[...,-1])
                        # plt.subplot(223), plt.imshow(data[...,4])
                        # plt.subplot(224), plt.imshow(data[...,5])
                        # plt.show()
                    # endregion

                # update indexes
                idx_j += int(img_size * (1-overlap))
            idx_i += int(img_size * (1-overlap))

        # region DEBUG - Decode graph encoding
        # keypoints_thr = 0.5
        # edges_thr = 0.5
        # kp_limit = 1000000000000
        # snap_dist = 100.0
        # angledistance_weight = 100
        # linestring_delta_meters = 20   # meters
        # decoding = decode_graph_encoding(
            # imagegraph,
            # max_degree=max_degree,
            # vector_norm=vector_norm,
            # keypoints_thr=keypoints_thr,
            # edges_thr=edges_thr,
            # kp_limit=kp_limit,
            # snap_dist=snap_dist,
            # angledistance_weight=angledistance_weight,
            # )
        # plot_graph(g), plt.show()
        # plt.subplot(121), plt.imshow(mask==1)
        # plt.subplot(122), plt.imshow(decoding)
        # plt.show()
        # endregion
