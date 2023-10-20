import os
import json
from pathlib import Path
import numpy as np
import random
import tifffile
from matplotlib import pyplot as plt

from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import Callback

import albumentations as A
import tensorflow as tf
from tensorflow.keras import backend as K


################################################################################################################################################
# Paths and dataset preparation
#
################################################################################################################################################
def get_training_paths(data_size=512, data_disk='D', data_ths: list[float] = None, data_paths: list[str] = None):
    """
    True data : Dataset with existing roads
    False data : Dataset with no roads
    """

    # Initialize train and test lists
    dataset_train, dataset_val = list(), list()
    
    # Get directions and data ratio
    data_ths = [] if data_ths is None else data_ths
    data_paths = [] if data_paths is None else data_paths  # TODO : Should be filled

    # Loop in folders
    for path, th in zip(data_paths, data_ths):
        for file in os.listdir(path):
            if np.random.random() <= th:
                if '/val/' in path:
                    dataset_val.append(path+file)
                else:
                    dataset_train.append(path+file)

    # dataset_train = random.sample(dataset_train, 60)
    # dataset_val = random.sample(dataset_val, 20)

    return dataset_train, dataset_val


################################################################################################################################################
# Data Generator
#
################################################################################################################################################
class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, data_paths, wanted_shape, batch_size, training_gen=True, data_scale='meanstd',):
        # Initialization
        self.wanted_shape = wanted_shape
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.shuffle = True
        self.training_gen = training_gen
        self.data_scale = data_scale
        
        # shuffle data
        self.on_epoch_end()
        
        # image augmentation
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.25, -0.1), p=0.25)
        ])
        
        # Set Data loader metadata
        set_file = Path(Path.cwd() / "settings.json")
        if not set_file.is_file():
            raise ValueError("The JSON file path provided is not a file.")
        settings = type('Settings', (object,), json.loads(set_file.read_text(encoding='utf-8')))()

        # Read metadata
        ge_metadata = np.load(settings.ge_metadata)
        self.ge_means = ge_metadata['means']
        self.ge_stds = ge_metadata['stds']
        wv2_metadata = np.load(settings.wv2_metadata)
        self.wv2_means = wv2_metadata['means']
        self.wv2_stds = wv2_metadata['stds']
        wv3_metadata = np.load(settings.wv3_metadata)
        self.wv3_means = wv3_metadata['means']
        self.wv3_stds = wv3_metadata['stds']
        wv4_metadata = np.load(settings.wv4_metadata)
        self.wv4_means = wv4_metadata['means']
        self.wv4_stds = wv4_metadata['stds']
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.data_paths) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_paths_temp = [self.data_paths[k] for k in indexes]

        # Generate data
        X, y = self._data_generation(list_paths_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def _data_generation(self, list_paths_temp):
        # Declare empty list
        X, Y = list(), list()
        
        for path in list_paths_temp:
            # Read image and pred
            img, mask = self.load_data_from_file(path)
            
            # Data augmentation
            if self.training_gen:
                img, mask = self.random_rotate_flip(img, mask)
            
            # scale data
            img = self.normalize_meanstd(img, path)

            img = cv2.resize(img, (self.wanted_shape[:-1], self.wanted_shape[:-1]))
            mask = cv2.resize(mask, (self.wanted_shape[:-1], self.wanted_shape[:-1]), interpolation=cv2.INTER_NEAREST)
            
            # Encode mask
            mask = np.expand_dims(np.float32(mask), -1)
            
            # Append to data
            X.append(np.copy(img))
            Y.append(np.copy(mask))
        
        # Prepare for tensorflow inputs
        X = np.asarray(X)
        Y = np.asarray(Y)

        return X, Y
    
    def load_data_from_file(self, data_path):
        data = tifffile.imread(data_path)
        img = data[...,:-1]
        mask = data[...,-1]
        return img, mask
    
    def normalize_meanstd(self, img, path):
        if '_GE01_' in path:
            means = self.ge_means
            stds = self.ge_stds
        elif '_WV02_' in path:
            means = self.wv2_means
            stds = self.wv2_stds
        elif '_WV03_' in path:
            means = self.wv3_means
            stds = self.wv3_stds
        elif '_WV04_' in path:
            means = self.wv4_means
            stds = self.wv4_stds
        return np.true_divide(np.subtract(img, means), stds)

    def random_rotate_flip(self, img, pred):
        # chose a random transformation
        rot_to_do = random.sample([None, 1, 2, 3], 1)[0]
        flip_to_do = random.sample([None, 0, 1], 1)[0]
        # do transformation on image and mask
        if rot_to_do is not None:
            img = np.rot90(img, k=rot_to_do, axes=(0,1))
            pred = np.rot90(pred, k=rot_to_do, axes=(0,1))
        if flip_to_do is not None:
            img = np.flip(img, flip_to_do)
            pred = np.flip(pred, flip_to_do)
        return img, pred


################################################################################################################################################
# Callbacks
#
#
################################################################################################################################################
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    learning_rate = 0.5 * learning_rate_base * (1 + np.cos(
        np.pi *
        (global_step - warmup_steps - hold_base_rate_steps
         ) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    if hold_base_rate_steps > 0:
        learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
        warmup_rate = slope * global_step + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


class WarmUpCosineDecayScheduler(Callback):
    """Cosine decay with warmup learning rate scheduler
    """

    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        """Constructor for cosine decay with warmup learning rate scheduler.
    Arguments:
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        global_step_init {int} -- initial global step, e.g. from previous checkpoint.
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
        verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose

    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        tf.summary.scalar('batch_lr', data=lr, step=self.global_step)

    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))
