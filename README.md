# 23-1016 RNCan
Project carried out for Natural Resources Canada to infer roads from satellite images and annotations.
____
## Solution structure
  ```
  23-1016_RNCan/
  │
  ├── aws_env.yml - metrics and data preparation dependencies
  │
  ├── dataset/ - Build dataset
  │   ├── get_sats_stats.py - compute means and standard deviation of satellites images
  │   ├── create_dataset.py - create dataset from raw data
  │   ├── clip_tiles_from_masks.py - clip tiles from georeferenced binary masks
  │   ├── create_graph_dataset.py - create graph to tensor encoding (GTE) dataset
  │   └── main.py - main script to start building of the dataset
  │
  ├── metrics/ - APLS and TOPO metrics
  │   ├── apls/ - APLS metric
  │   ├── topo/ - TOPO metric
  │   ├── other_metrics.py - segmentation evaluation metrics imported from scikit-learn
  │   └── main.py - main script to start evaluation
  │
  ├── postprocess/ - Postprocess predictions
  │   ├── main.py - main script to start postprocessing on batch
  │   ├── prob_to_masks.py - binary thresholding on models outputs
  │   ├── process_gte.py - process GTE
  │   └── process_prediction.py - custom process of roads inference
  │
  └── training/ - Training and inference
     ├── train_cnn_tensorflow/ - CNN training and inference with Tensorflow
     │   ├── keras_unet_collection/ - tensorflow models and utilities implementation
     │   └── train.py - Run training process for Tensorflow scenario
     ├── train_cnn_pytorch/ - CNN training and inference with Pytorch
     │   ├── settings.json - Providers metadata
     │   └── main.py - Run training process for PyTorch scenario
     ├── train_gte_pytorch/ - GTE training and inference with Pytorch
     │    ├── settings.json - Providers metadata
     │    └── main.py - run training process for PyTorch and GTE scenario
     ├── tf_env.yml - training and inference dependencies using Tensorflow
     └── pt_env.yml - training and inference dependencies using Pytorch
  ```

____
## Dependencies
All dependencies for data preparation and evaluation can be found in conda output [aws_env.yml](aws_env.yml)

All dependencies for training execution can be found in :
* PyTorch: [training/pt_env.yml](training/pt_env.yml)
* TensorFlow: [training/tf_env.yml](training/tf_env.yml)

____
## License
Current work published under [MIT LICENSE](LICENSE.txt)

For metrics, see inherited [LICENSE](metrics/LICENSE.txt) from CosmiQ project.

----
## Theory & repository

### Train CCN PyTorch
PyTorch Implementation of U-Net, R2U-Net, Attention U-Net, Attention R2U-Net
https://github.com/LeeJunHyun/Image_Segmentation

**U-Net: Convolutional Networks for Biomedical Image Segmentation**

https://arxiv.org/abs/1505.04597

**Recurrent Residual Convolutional Neural Network based on U-Net (R2U-Net) for Medical Image Segmentation**

https://arxiv.org/abs/1802.06955

**Attention U-Net: Learning Where to Look for the Pancreas**

https://arxiv.org/abs/1804.03999

**Attention R2U-Net: Just integration of two recent advanced works (R2U-Net + Attention U-Net)**

### Train CNN Tensorflow
keras-unet-collection based on Yingkai Sha - v0.0.13: https://github.com/yingkaisha/keras-unet-collection.

### Metrics
APLS and TOPO metrics calculation based on APLS Metric from CosmisQ works. Original repository is available [here](https://github.com/CosmiQ/apls).

Framework is mostly based on paper from :

*Biagioni, J. & Eriksson, J. Inferring Road Maps from Global Positioning System Traces: Survey and Comparative Evaluation. Transportation Research Record 2291, 61–71 (2012).*

____
## Usage

### 1. Dataset
*main.py* for building dataset. 

    python dataset/main.py
            -save_dir C:/User/MyUser/MyProject/data/Datasets
            -data_dir C:/User/MyUser/MyProject/data/TIFFS
            -annotations_index_path C:/User/MyUser/MyProject/data/index_annotations_final.gpkg
            -annotations_dir C:/User/MyUser/MyProject/data/annotations
            -tile_mask_dir C:/User/MyUser/MyProject/data/Tiles
            -tile_save_dir C:/User/MyUser/MyProject/data/Masks
            -graph_gt C:/User/MyUser/MyProject/data/Annotations/Tiles
            -graph_save_dir C:/User/MyUser/MyProject/data/Datasets

Process will :
1. Compute means and standard deviation of satellites images;
2. Create binary masks dataset from annotations (polyLine);
3. Clip tiles from masks rasters to save memory during step 4;
4. Create GTE dataset from tiled masks.

Take into account that annotations have been rasterized and transformed into graph before step 3. Rasterization process is located in the [prob_to_masks.py](postprocess/prob_to_masks.py) file of the _postprocess_ module and graph transformation process is located in the metrics folder following the skeletonization. Each step can be executed separately by using the execute function. The _train_val_zones.npz_ file, used to split the dataset into training and validation zones, is provided in the _dataset_ module.

#### Available parameters
* save_dir (required): Global save directory path
* data_dir (required): Global data directory path
* annotations_index_path (required): Annotations index file path
* annotations_dir (required): Annotations directory path
* tile_mask_dir (required): Tile mask directory path
* tile_save_dir (required): Tile save directory path
* graph_gt (required): Mask and graph ground truth directory path
* graph_save_dir (required): Graph save directory path
* stats_output_extension: Get file name from index
* gs: Providers list
* gb: Raster bands list
* tile_size: Tile size in pixels
* tile_overlap: Tile overlaping when clipping, in pixels
* graph_image_size: Size if a training clip in pixels
* graph_image_overlap: Training clip overlap in %
* graph_encoding_max_degree: Maximum degrees of freedom in which edges’ angles and will be encoded
* graph_encoding_r: Norm of the vector between 2 nodes
* graph_linestring_delta_meters: Maximum distance between 2 nodes for nodes injection

### 2. Training

#### 2.1 PyTorch
*main.py* to start training.

    python training/train_cnn_pytorch/main.py
            --log_path C:/User/MyUser/MyProject/training/logs/
            --exp_name pytorch_compactgte_loss100_noinit_lr5e5_fulldataset
            --model_path C:/User/MyUser/MyProject/Models/
            --train_path C:/User/MyUser/MyProject/data/train/
            --valid_path C:/User/MyUser/MyProject/data/val/

##### Available parameters
* log_path (required): Log directory path
* exp_name (required): Experiment name
* model_path (required): Model directory path
* train_path (required): Training dataset directory path
* valid_path (required): Validation dataset directory path
* image_size: Size if a training clip in pixels
* t: Number of convolutions on recurrent convolutional blocks
* img_ch: Number of image channels
* output_ch: Number of channels for output
* num_epochs: Number of epochs
* num_epochs_decay: Number of epochs for decaying learning rate
* batch_size: Batch size
* num_workers: Number of workers for data loader
* lr: Learning rate
* beta1: Beta1 for Adam optimizer
* beta2: Beta2 for Adam optimizer
* mode: Execution mode - only train is available
* model_type: Type of architecture to train
* cuda_idx: Id of gpu to use for training

##### settings.json
[settings.json](training/train_cnn_pytorch/settings.json) used for providers metadata.
    
    {
      "ge_metadata": "C:/User/MyUser/MyProject/data/Datasets/geoeye-1-ortho-pansharp_metadata.npz",
      "wv2_metadata": "C:/User/MyUser/MyProject/data/Datasets/worldview-2-ortho-pansharp_metadata.npz",
      "wv3_metadata": "C:/User/MyUser/MyProject/data/Datasets/worldview-3-ortho-pansharp_metadata.npz",
      "wv4_metadata": "C:/User/MyUser/MyProject/data/Datasets/worldview-4-ortho-pansharp_metadata.npz"
    }

#### 2.2 PyTorch GTE
See the _2.1 PyTorch_ section for parameters and use case. Adjust the variables and paths.

#### 2.3 Tensorflow
*train.py* to start training.

    python training/train_cnn_pytorch/main.py
            --models_dir C:/User/MyUser/MyProject/training/logs/
            --logs_dir C:/User/MyUser/Models/

##### Available parameters
* models_dir (required): Models directory path
* logs_dir (required): Logs directory path
* model_name: Type of architecture to train
* exp_name: Experiment name
* n_epochs: Number of epochs
* n_batch: Batch size

### 3. Postprocess
*main.py* postprocess predictions from CNN models. It will process CNN probability output to create binary masks.

    python postprocess/main.py
        -index_path C:/User/MyUser/MyProject/data/index_annotations_final.gpkg \
        -postprocess_gte True \
        -gte_pred_dir C:/User/MyUser/MyProject/data/R2AUNet_gte \
        -gte_save_dir C:/User/MyUser/MyProject/data/Comparison_methods/R2AUnet_gte/ \
        -postprocess_pred True \
        -pred_model_name R2AUnet_pt \
        -pred_dir C:/User/MyUser/MyProject/data/postprocess/ \
        -pred_save_folder postprocessed_pt \
        -pred_save_dir C:/User/MyUser/MyProject/data/Comparison_methods/ \
        -create_masks True \
        -mask_model_name R2AUnet_pt \
        -mask_save_dir C:/User/MyUser/MyProject/data/postprocess/Comparison_methods/ \
        -mask_gt C:/User/MyUser/MyProject/data/GT/annotations/ \
        -mask_pred_dir C:/User/MyUser/MyProject/data/postprocess/

Each step can be executed separately by using the execute function.

**Warning**: To optimize storage efficiency, we've limited predictions to match the minimum annotation extent. This limitation is implemented using the rasterio library, which introduces a 0.5 pixel offset during each clipping operation and rounds it to 0 after the first iteration. If you're working with an already clipped raster, avoid re-clipping it to prevent the addition of a 0.5 offset, which could result in a total 1-pixel offset.

#### Available parameters
* index_path (required): Annotations index file path
* postprocess_gte: Postprocess GTE should be executed
* gte_pred_dir: GTE prediction directory path
* gte_save_dir: GTE postprocess results save directory path
* postprocess_pred: Postprocess prediction should be executed
* pred_model_name: Prediction model name
* pred_dir: Prediction directory path
* pred_save_folder: Prediction postprocess results save folder name
* pred_save_dir: Prediction postprocess results save directory path
* create_masks: Binary thresholding on models outputs should be executed
* mask_model_name: Model name
* mask_save_dir: Mask save directory path
* mask_gt: Mask ground truth directory path
* mask_pred_dir: Mask prediction directory path

### 4. Metrics
*main.py* runs calculation for:
1. Skeletonize predictions;
2. Graph transformation and georeferencement;
3. Compute APLS metric;
4. Compute TOPO metric;
5. Compute segmentation evaluation metric.
	

	python metrics/main.py 
        -comp_root_dir C:/User/MyUser/MyProject/data/Comparison_methods/ \

#### Available parameters
* comp_root_dir (required): Truth directory relative path

#### Outputs
CSV report file with the following columns:
* outroot: Absolute path of the graph file
* APLS: APLS score
* APLS_gt_onto_prop: APLS score of ground truth onto proposal
* APLS_prop_onto_gt: APLS score of proposal onto ground truth
* accuracy: Classification accuracy score
* precision: Classification precision score
* f1_score: Classification F1 score, also known as balanced F-score or F-measure
* topo_recall: TOPO metric recall score
* topo_prec: TOPO metric precision score
* topo_f1: TOPO metric F1 score

____
## TODO
* [ ] Add travel time and speed key