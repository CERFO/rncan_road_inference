import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'  # GPU selection

import argparse
import numpy as np
from matplotlib import pyplot as plt
from models import att_r2_unet, get_coswin_model
from utils import get_training_paths, DataGenerator, WarmUpCosineDecayScheduler
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
import tensorflow as tf


def main(config):
    # Training parameters
    IMG_SHAPE = (128, 128, 4)

    # Get datasets paths and generators
    dataset_train, dataset_val = get_training_paths()
    training_generator = DataGenerator(dataset_train, IMG_SHAPE, config.n_batch)
    validation_generator = DataGenerator(dataset_val, IMG_SHAPE, config.n_batch, training_gen=False)

    # Warm up cosine decay parameters
    warmup_learning_rate = 0.0
    warmup_epoch = 6
    hold_base_rate_epoch = 14
    sample_count = len(dataset_train)
    total_steps = int(config.n_epochs * sample_count / config.n_batch)
    warmup_steps = int(warmup_epoch * sample_count / config.n_batch)
    hold_base_rate_steps = int(hold_base_rate_epoch * sample_count / config.n_batch)

    # Get model
    if config.model_name == 'attr2unet':
        model = att_r2_unet(IMG_SHAPE, n_labels=1, lr=1e-6)
        learning_rate_base = 7e-4
    elif config.model_name == 'coswin':
        model = get_coswin_model(IMG_SHAPE, n_labels=1)
        learning_rate_base = 7e-4

    model.summary()
    print('Dataset train :', len(dataset_train), ' files')
    print('Dataset val :', len(dataset_val), ' files')

    # Model callbacks
    lr_scheduler = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_base, total_steps=total_steps
                                              , warmup_learning_rate=warmup_learning_rate, warmup_steps=warmup_steps
                                              , hold_base_rate_steps=hold_base_rate_steps)
    save_to_csv = CSVLogger(config.logs_dir + config.exp_name + '.log', append=True)
    model_checkpoint = ModelCheckpoint(config.models_dir + config.exp_name + '.hdf5', save_weights_only=False)

    # Test image
    # For testing - user validation of the dataset
    ix = np.random.randint(0, len(dataset_train), 10)
    val_to_get = [dataset_train[i] for i in ix]
    X_test, Y_test = training_generator._data_generation(val_to_get)
    print('Input  [min, max, shape] :', X_test.min(), ',', X_test.max(), ',', X_test.shape)
    print('Target [min, max, shape] :', Y_test.min(), ',', Y_test.max(), ',', Y_test.shape)
    for i in range(X_test.shape[0]):
        plt.subplot(131), plt.imshow(X_test[i,...,0])
        plt.subplot(132), plt.imshow(X_test[i,...,-1])
        plt.subplot(133), plt.imshow(Y_test[i], clim=[0,1])
        plt.show()

    # Train model
    model.fit(x=training_generator,
              validation_data=validation_generator,
              epochs=config.n_epochs,
              verbose=2,
              max_queue_size=4*config.n_batch,
              workers=24,
              callbacks=[lr_scheduler, save_to_csv, model_checkpoint]
              )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Misc
    parser.add_argument('--models_dir', type=str, required=True, default=None)
    parser.add_argument('--logs_dir', type=str, required=True, default=None)
    parser.add_argument('--model_name', type=str, default='attr2unet')
    parser.add_argument('--exp_name', type=str, default='_tf_baseline')

    # training hyper-parameters
    parser.add_argument('--n_epochs', type=int, default=75)
    parser.add_argument('--n_batch', type=int, default=20)

    config = parser.parse_args()

    main(config)
