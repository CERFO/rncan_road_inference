# 23-1016 RNCan
Project carried out for Natural Resources Canada to infer roads from satellite images and annotations.
____
## Solution structure
  ```
  23-1016_RNCan/
  │
  ├── environment.yml - metrics and data preparation dependencies
  ├── models_tf_environment.yml - training and inference dependencies using Tensorflow
  ├── models_pt_environment.yml - training and inference dependencies using Pytorch
  │
  ├── dataset/ - Build dataset
  │   ├── get_sats_stats.py - retrieve raw data from repository
  │   ├── create_dataset.py - create dataset from raw data
  │   ├── clip_tiles_from_masks.py - clip tiles from masks
  │   ├── create_graph_dataset.py - create graph dataset
  │   └── main.py - main script to start building of the dataset
  │
  ├── metrics/ - APLS and TOPO metrics
  │   ├── apls/ - APLS metrics
  │   ├── topo/ - TOPO metrics
  │   └── main.py - main script to start evaluation
  │
  ├── postprocess/ - Postprocess predictions
  │   ├── main.py - main script to start postprocessing on batch
  │   ├── create_masks_for_comp.py - create masks for comparison
  │   ├── process_gte.py - process GTE
  │   └── process_prediction.py - process prediction from PT and TF
  │
  └── training/ - Training and inference
     ├── train_cnn_tensorflow/ - CNN training and inference with Tensorflow
     │   ├── keras_unet_collection/ - The tensorflow.keras models implementation
     │   └── train.py - Run training process for Tensorflow scenario
     ├── train_cnn_pytorch/ - CNN training and inference with Pytorch
     │   ├── settings.json - Providers metadata
     │   └── main.py - Run training process for PyTorch scenario
     └── train_gte_pytorch/ - GTE training and inference with Pytorch
         ├── settings.json - Providers metadata
         └── main.py - run training process for PyTorch and GTE scenario
  ```

____
## Dependencies
All dependencies for data preparation and evaluation can be found in conda output [environment.yml](environment.yml)

All dependencies for training execution can be found in :
* PyTorch : [training/pt_environment.yml](training/pt_environment.yml)
* TensorFlow : [training/tf_environment.yml](training/tf_environment.yml)

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

**Attention R2U-Net : Just integration of two recent advanced works (R2U-Net + Attention U-Net)**

### Train CNN Tensorflow
keras-unet-collection based on Yingkai Sha - v0.0.13 : https://github.com/yingkaisha/keras-unet-collection.

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
            -get_file_name _metadata.npz
            -annotations_index_path C:/User/MyUser/MyProject/data/index_annotations_final.gpkg
            -annotations_dir C:/User/MyUser/MyProject/data/annotations
            -tile_mask_dir C:/User/MyUser/MyProject/data/Tiles
            -tile_save_dir C:/User/MyUser/MyProject/data/Masks
            -graph_gt C:/User/MyUser/MyProject/data/Annotations/Tiles
            -graph_save_dir C:/User/MyUser/MyProject/data/Datasets

Process will :
1. Retrieve and prepare raw data from repository
2. Generate binary masks from annotations
3. Clip images from masks
4. Create graphs

Take into account that annotations have been rasterized - rasterization process is located in the _create_mask_for_comp.py_ file of the _postprocess_ module. Each step can be executed separately by using the execute function. The _train_val_zones.npz_ file, used to split the dataset into training and validation zones, is provided in the _dataset_ module.

#### Available parameters
* save_dir (required) : Global save directory path
* data_dir (required) : Global data directory path
* get_file_name (required) : Get file name from index
* annotations_index_path (required) : Annotations index file path
* annotations_dir (required) : Annotations directory path
* tile_mask_dir (required) : Tile mask directory path
* tile_save_dir (required) : Tile save directory path
* graph_gt (required) : Graph ground truth directory path
* graph_save_dir (required) : Graph save directory path
* gs : Providers list
* gb : Raster bands list
* tile_size : Tile size relative to CRS
* tile_overlap : Tile overlap relative to CRS
* graph_image_size : Graph image size relative to CRS
* graph_image_overlap : Graph image overlap relative to CRS
* graph_encoding_max : Graph encoding max value
* graph_encoding_r
* graph_linestring_delta_meters

### 2. Training

#### 2.1 PyTorch
*main.py* to start training.

    python training/train_cnn_pytorch/main.py
            --log_path C:/User/MyUser/MyProject/training/logs/
            --exp_name pytorch_compactgte_loss100_noinit_lr5e5_fulldataset
            --model_path C:/User/MyUser/Datasets/compact_gte_delta20_vnorm40/train/true_data/
            --train_path C:/User/MyUser/Datasets/compact_gte_delta20_vnorm40/val/true_data/
            --valid_path C:/User/MyUser/Datasets/compact_gte_delta20_vnorm40/val/true_data/

##### Available parameters
* log_path (required) : Log directory path
* exp_name (required) : Experiment name
* model_path (required) : Model directory path
* train_path (required) : Training dataset directory path
* valid_path (required) : Validation dataset directory path
* image_size : Image size relative to CRS
* t : t for Recurrent step of R2U_Net or R2AttU_Net
* img_ch : Number of image channels
* output_ch : Number of channels for output
* num_epochs : Number of epochs
* num_epochs_decay : Number of epochs for decaying learning rate
* batch_size : Batch size
* num_workers : Number of workers for data loader
* lr : Learning rate
* beta1 : Beta1 for Adam optimizer
* beta2 : Beta2 for Adam optimizer
* augmentation_prob
* log_step
* val_step
* mode
* model_type
* cuda_idx

##### settings.json
_settings.json_ used for providers metadata.
    
    {
      "ge_metadata": "C:/User/MyUser/MyProject/data/Datasets/geoeye-1-ortho-pansharp_metadata.npz",
      "wv2_metadata": "C:/User/MyUser/MyProject/data/Datasets/worldview-2-ortho-pansharp_metadata.npz",
      "wv3_metadata": "C:/User/MyUser/MyProject/data/Datasets/worldview-3-ortho-pansharp_metadata.npz",
      "wv4_metadata": "C:/User/MyUser/MyProject/data/Datasets/worldview-4-ortho-pansharp_metadata.npz"
    }

##### 2.2 PyTorch GTE
See the _2.1 PyTorch_ section for parameters and use case. Adjust the variables and paths.

#### 2.2 Tensorflow
*train.py* to start training.

    python training/train_cnn_pytorch/main.py
            --models_dir C:/User/MyUser/MyProject/training/logs/
            --logs_dir C:/User/MyUser/Models/

##### Available parameters
* models_dir (required) : Models directory path
* logs_dir (required) : Logs directory path
* model_name : Model name
* exp_name : Experiment name
* n_epochs : Number of epochs
* n_batch : Batch size

### 3. Postprocess
*main.py* postprocess predictions from CNN models. It will create masks for comparison and process GTE.

    python postprocess/main.py
        -index_path C:/User/MyUser/MyProject/data/index_annotations_final.gpkg \
        -postprocess_gte True \
        -gte_pred_dir C:/User/MyUser/MyProject/data/R2AUNet_gte \
        -gte_save_dir C:/User/MyUser/MyProject/data/Comparaison_methodes/R2AUnet_gte/ \
        -postprocess_pred True \
        -pred_model_name R2AUnet_pt \
        -pred_dir C:/User/MyUser/MyProject/data/postprocess/ \
        -pred_save_folder postprocessed_pt \
        -pred_save_dir C:/User/MyUser/MyProject/data/Comparaison_methodes/ \
        -create_masks True \
        -mask_model_name R2AUnet_pt \
        -mask_save_dir C:/User/MyUser/MyProject/data/postprocess/Comparaison_methodes/ \
        -mask_gt C:/User/MyUser/MyProject/data/GT/annotations/ \
        -mask_pred_dir C:/User/MyUser/MyProject/data/postprocess/

Each step can be executed separately by using the execute function.

#### Available parameters
* index_path (required) : Index file path
* postprocess_gte : Postprocess GTE should be executed
* gte_pred_dir : GTE prediction directory path
* gte_save_dir : GTE postprocess results save directory path
* postprocess_pred : Postprocess prediction should be executed
* pred_model_name : Prediction model name
* pred_dir : Prediction directory path
* pred_save_folder : Prediction postprocess results save folder name
* pred_save_dir : Prediction postprocess results save directory path
* create_masks : Create masks for comparison should be executed
* mask_model_name : Model name
* mask_save_dir : Mask save directory path
* mask_gt : Mask ground truth directory path
* mask_pred_dir : Mask prediction directory path

### 4. Metrics
*main.py* runs calculation for : 
1. Skeletonize predictions
2. Graph transformation and georeferencement
3. Compute APLS metric
4. Compute TOPO metric
5. Compute other metrics based on ground truth / proposal
	

	python metrics/main.py 
        -comp_root_dir C:/User/MyUser/MyProject/data/Comparaison_methodes/ \

#### Available parameters
* comp_root_dir (required) : Truth directory relative path

#### Outputs
CSV report file with the following columns:
* outroot : Absolute path of the graph file
* APLS : APLS score
* APLS_gt_onto_prop : APLS score of ground truth onto proposal
* APLS_prop_onto_gt : APLS score of proposal onto ground truth
* accuracy : Accuracy score - truth / proposal
* precision : Precision score - truth / proposal
* f1_score : F1 score - truth / proposal
* topo_recall : TOPO metric recall
* topo_prec : TOPO metric precision
* topo_f1 : TOPO metric f1 score

____
## TODO
* [ ] Add travel time and speed key