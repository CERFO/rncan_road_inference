import argparse
import os
from subprocess import call
from tqdm import tqdm
from pathlib import Path
import other_metrics as om


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Root directory
    parser.add_argument('-comp_root_dir', action='store', dest='comp_root_dir', type=str,
                        help='Root dir as path', required=True)
    
    args = parser.parse_args()
    comp_root_dir = args.comp_root_dir

    ######################################################
    # Skeletonize predictions
    #   - pred_mask_dir : directory that contains GeoTiffs road masks
    #   - results_dir   : directory to write output skeleton file
    ######################################################
    masks_folder = [folder for folder in os.listdir(comp_root_dir) if os.path.isdir(comp_root_dir+folder)]

    for mask_folder in tqdm(masks_folder):
        call(['python', 'apls/skeletonize.py',
                '--pred_mask_dir', comp_root_dir+mask_folder,
                '--results_dir', comp_root_dir+mask_folder,
                '--n_threads', '1',
            ])
    
    ######################################################
    # Graph transformation and georeferencement
    #   - pred_mask_dir : directory that contains GeoTiffs road masks
    #   - wkt_csv_file  : directory that contains skeleton wkt file
    #   - results_dir   : directory to write output georeferenced graph
    ######################################################
    masks_folder = [folder for folder in os.listdir(comp_root_dir) if os.path.isdir(comp_root_dir+folder)]

    for mask_folder in tqdm(masks_folder):
        call(['python', 'apls/wkt_to_G.py',
                '--pred_mask_dir', comp_root_dir+mask_folder,
                '--wkt_csv_file', comp_root_dir+mask_folder+'/wkt_nospeed.csv',  # Output file
                '--results_dir', comp_root_dir+mask_folder,
            ])
    
    ######################################################
    # Compute APLS
    #   - output_dir    : directory to write results (csv)
    #   - truth_dir     : directory that contains ground truth graphs
    #   - prop_dir      : directory that contains predicted graphs
    ######################################################
    masks_folder = [folder for folder in os.listdir(comp_root_dir) if os.path.isdir(comp_root_dir + folder) and folder != 'GT']

    for mask_folder in tqdm(masks_folder):
        call(['python', 'apls/apls.py',
                '--output_dir', comp_root_dir,
                '--truth_dir', comp_root_dir + '/GT',
                '--prop_dir', comp_root_dir + '/' + mask_folder,
                '--n_threads', '1',
              ])
    
    ######################################################
    # Compute TOPO
    #   - truth_dir    : Directory that contains ground truth graphs
    #   - prop_dir     : Directory that contains predicted graphs
    #   - output_path  : Directory that contains output report
    ######################################################
    comp_root_path = Path(comp_root_dir)
    comp_sub_dirs = list(comp_root_path.iterdir())
    masks_folder = [folder.name for folder in comp_sub_dirs if folder.is_dir() and folder.name != 'GT']

    for mf in masks_folder:
        call(['python', 'topo/main.py',
                '-truth_dir', Path(comp_root_path, 'GT'),
                '-prop_dir', Path(comp_root_path, mf),
                '-output_path', comp_root_path
            ])
    
    ######################################################
    # Compute other metrics
    #   - truth_dir     : directory that contains ground truth masks
    #   - prop_dir      : directory that contains predicted masks
    #   - results_dir   : directory to the results csv
    ######################################################
    for masks_folder in tqdm(os.listdir(comp_root_dir)):
        if masks_folder == 'GT' or not os.path.isdir(comp_root_dir+masks_folder):
            continue

        om.compute_metrics(
                truth_dir = comp_root_dir+'/GT',
                prop_dir = comp_root_dir+'/'+masks_folder,
                results_dir = comp_root_dir+masks_folder+'.csv',
            )
