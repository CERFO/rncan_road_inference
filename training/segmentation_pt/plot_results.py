import numpy as np
from matplotlib import pyplot as plt


def execute(paths: list[str], exp_names: list[str] = None, to_monitor: list[str] = None):
    # Config
    paths = paths
    exp_names = ['R2AUnet_pytorch','R2AUnet_tensorflow'] if exp_names is None else exp_names
    to_monitor = ['loss', 'val_loss', 'val_DiceCoef'] if to_monitor is None else to_monitor

    # Plot
    for i, monitor in enumerate(to_monitor):
        legend_names = list()

        # create subplot
        plt.subplot(1, len(to_monitor), i+1)
        plt.title(monitor)
        plt.xlabel('epochs')

        for path, exp_name in zip(paths,exp_names):
            # get column's index to load from file
            monitors = np.loadtxt(path+'.log', dtype=str, delimiter=',', max_rows=1)
            idx = np.where(monitors == monitor)[0]

            # read and plot results if monitored
            if idx.size != 0:
                results = np.loadtxt(path+'.log', delimiter=',', skiprows=1, usecols=idx)
                plt.plot(results)
                legend_names.append(exp_name)

        # add legend to subplot
        plt.legend(legend_names)

    plt.show()
