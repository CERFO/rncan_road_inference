import os


def compute_metrics(truth_dir: str, prop_dir: str, results_dir: str) -> None:
    from osgeo import gdal
    import rasterio
    import pandas
    import numpy as np
    from sklearn.metrics import precision_score, accuracy_score, f1_score

    # Get results dataframe and initilize metrics
    df = pandas.read_csv(results_dir)
    df['accuracy'] = 0
    df['precision'] = 0
    df['f1_score'] = 0

    # Loop in masks files
    for file in os.listdir(truth_dir):
        if not file.endswith('.tif'):
            continue
        if not (df['outroot'].str.contains(file[:-9]) == True).any():
            continue

        # Open files
        with rasterio.open(os.path.join(truth_dir, file)) as truth_file, rasterio.open(
                os.path.join(prop_dir, file)) as prop_file:
            truth_mask = truth_file.read()[0]
            prop_mask = prop_file.read()[0]
            nodata_val = prop_file.profile['nodata']

        # Flatten arrays and remove nodata
        truth_mask = truth_mask.flatten()
        truth_mask = np.delete(truth_mask, truth_mask == nodata_val)
        prop_mask = prop_mask.flatten()
        prop_mask = np.delete(prop_mask, prop_mask == nodata_val)

        # Compute metrics
        precision = precision_score(truth_mask, prop_mask)
        acc = accuracy_score(truth_mask, prop_mask)
        fscore = f1_score(truth_mask, prop_mask)

        # Write to csv
        df.loc[df['outroot'].str.contains(file[:-9]), 'accuracy'] = acc
        df.loc[df['outroot'].str.contains(file[:-9]), 'precision'] = precision
        df.loc[df['outroot'].str.contains(file[:-9]), 'f1_score'] = fscore

    # Save results csv
    df.to_csv(results_dir)