import numpy as np
from matplotlib import pyplot as plt


def execute(log_dir: str, exp_names: list[str] = None, to_monitor: list[str] = None):
    # Config
    exp_names = [
        'pytorch_gte_lr1e4_compact_loss100_noinit',
        'pytorch_gte_lr1e4_compact_loss100_noinit_lr5e5',
        'pytorch_gte_lr1e4_compact_loss100_noinit_lr5e5_fulldataset',
        'pytorch_compactgte_loss100_noinit_lr5e5_fulldataset',
        ] if exp_names is None else exp_names
    to_monitor = [
        'train_epoch_loss',
        'train_keypoint_prob_loss',
        'train_direction_prob_loss',
        'train_direction_vector_loss',
        'val_epoch_loss',
        'val_keypoint_prob_loss',
        'val_direction_prob_loss',
        'val_direction_vector_loss',
        ] if to_monitor is None else to_monitor

    # Plot
    for i, monitor in enumerate(to_monitor):
        legend_names = list()

        # create subplot
        plt.subplot(2, len(to_monitor)//2, i+1)
        plt.title(monitor)
        plt.xlabel('epochs')

        for exp_name in exp_names:
            # get column's index to load from file
            monitors = np.loadtxt(log_dir+exp_name+'.log', dtype=str, delimiter=',', max_rows=1)
            idx = np.where(monitors == monitor)[0]

            # read and plot results if monitored
            if idx.size != 0:
                idx = idx+1 if exp_name == 'pytorch_gte_lr1e4' else idx
                results = np.loadtxt(log_dir+exp_name+'.log', delimiter=',', skiprows=1, usecols=idx)
                plt.plot(results)

                if exp_name == 'pytorch_gte_lr1e4_compact_loss100_noinit':
                    legend_name = 'dev_dataset_lr1e-4'
                elif exp_name == 'pytorch_gte_lr1e4_compact_loss100_noinit_lr5e5':
                    legend_name = 'dev_dataset_lr5e-5'
                elif exp_name == 'pytorch_gte_lr1e4_compact_loss100_noinit_lr5e5_fulldataset':
                    legend_name = 'full_dataset_lr5e-5'
                elif exp_name == 'pytorch_compactgte_loss100_noinit_lr5e5_fulldataset':
                    legend_name = 'full_dataset_lr5e-5_30epochs'
                legend_names.append(legend_name)

        # add legend to subplot
        plt.legend(legend_names)

    plt.show()