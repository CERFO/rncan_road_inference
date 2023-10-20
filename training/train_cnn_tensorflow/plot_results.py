import numpy as np
from matplotlib import pyplot as plt


def execute(project_path: str, paths: dict = None, to_monitor: list = None):
    # Config
    paths = {'attr2unet_tf_baseline'} if paths is None else paths
    to_monitor = ['loss', 'val_loss', 'val_DiceCoef'] if to_monitor is None else to_monitor

    # Plot
    for i, monitor in enumerate(to_monitor):
        legend_names = list()
        # create subplot
        plt.subplot(1, len(to_monitor), i+1)
        plt.title(monitor)
        plt.xlabel('epochs')

        for path in paths:
            # get column's index to load from file
            monitors = np.loadtxt(project_path+path+'.log', dtype=str, delimiter=',', max_rows=1)
            if path == 'attr2unet_320_cloud_B15911' and monitor == 'val_DiceCoef':
                monitor_ = 'val_cat_DiceCoef'
            else:
                monitor_ = monitor
            idx = np.where(monitors == monitor_)[0]

            # read and plot results if monitored
            if idx.size != 0:
                results = np.loadtxt(project_path+path+'.log', delimiter=',', skiprows=1, usecols=idx)
                plt.plot(results)
                legend_names.append(path)

        # add legend to subplot
        plt.legend(legend_names)

    plt.show()
