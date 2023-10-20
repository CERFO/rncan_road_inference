import os
from utils import *
from tqdm import tqdm
from rasterio import windows

def execute(annotations_index_path: str, pred_dir: str, save_dir: str
            , max_degree: int=6, vector_norm: int=40, line_thickness: int=10, kp_limit: int=1e12
            , keypoints_thr: float=0.1 , edges_thr: float=0.08, candidate_search_dist: float=80.0, snap_dist: float=200.0, angledistance_weight: float=50.0
            , max_tiff_size: int=40e9, nodata_val: int=255, load_size: int=5000, overlap: float=0.1):

    if not os.path.exists(annotations_index_path):
        raise ValueError('annotations_index_path does not exist.')
    if not os.path.exists(pred_dir):
        raise ValueError('pred_dir does not exist.')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Loop in prediction folder
    for file in tqdm(os.listdir(pred_dir)):

        print("Processing {file}...".format(file=file))

        # Get file name
        pred_path = pred_dir + file
        save_path = save_dir + '_'.join(file.split('_')[:-3]) + '_mask.tif'
        
        # GTE decoding
        if os.path.getsize(pred_dir + file) < max_tiff_size:

            # Read compact GTE prediction
            with rasterio.open(pred_path) as pred_file:
                compact_gte = pred_file.read()
                kwds = pred_file.profile
            
            # Prepare GTE for decoding
            decoding = prepare_gte(compact_gte, max_degree)
            del compact_gte
            
            # Decode graph encoding
            decoding = decode_graph_encoding(
                decoding,
                max_degree=max_degree,
                vector_norm=vector_norm,
                keypoints_thr=keypoints_thr,
                edges_thr=edges_thr,
                kp_limit=kp_limit,
                snap_dist=snap_dist,
                angledistance_weight=angledistance_weight,
                line_thickness=line_thickness,
                )
            
            # Save raster
            decoding[decoding>1] = 1
            with rasterio.open(save_path, 'w', **kwds) as save_file:
                save_file.write_band(1, decoding)
        
        else:
            # Open prediction file
            pred_file = rasterio.open(pred_path)
            
            # Open save file
            pred_kwds = pred_file.profile
            pred_kwds['count'] = 1
            pred_kwds['dtype'] = np.uint8
            save_file = rasterio.open(save_path, 'w+', **pred_kwds)
            
            # Loop in file and process
            idx_i = 0
            last_i, done_i = False, False
            while not done_i:
                idx_j = 0
                last_j, done_j = False, False
                while not done_j:
                    
                    # read prediction
                    win = windows.Window(idx_j, idx_i, load_size, load_size)
                    compact_gte = pred_file.read(window=win)
                    
                    # Prepare GTE for decoding
                    decoding = prepare_gte(compact_gte, max_degree)
                    del compact_gte
                    
                    # Decode graph encoding
                    decoding = decode_graph_encoding(
                        decoding,
                        max_degree=max_degree,
                        vector_norm=vector_norm,
                        keypoints_thr=keypoints_thr,
                        edges_thr=edges_thr,
                        kp_limit=kp_limit,
                        snap_dist=snap_dist,
                        angledistance_weight=angledistance_weight,
                        line_thickness=line_thickness,
                        )
                    
                    # save to file
                    current_save = save_file.read(window=win)[0]
                    decoding[current_save==1] = 1
                    decoding[decoding>1] = 1
                    save_file.write_band(1, decoding, window=win)
                    
                    # calculate new idx_j
                    idx_j, last_j, done_j = calculate_idx(idx_j, last_j, pred_file.shape[1], overlap, load_size)
                # calculate new idx_i
                idx_i, last_i, done_i = calculate_idx(idx_i, last_i, pred_file.shape[0], overlap, load_size)
            save_file.close()

        # Clip to minimum annotated zone for comparison
        with rasterio.open(save_path) as save_file:
            decoding = save_file.read()[0]
            kwds = save_file.profile
            pred_crs = pred_file.crs
        decoding, kwds = clip_to_min_extent(annotations_index_path, decoding, file, pred_crs, kwds, nodata_val)
        with rasterio.open(save_path, 'w', **kwds) as save_file:
            save_file.write_band(1, decoding)