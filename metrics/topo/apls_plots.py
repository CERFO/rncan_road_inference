# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:34:07 2019
@author: avanetten

Updated on Mon Sep 11 09:00:00 2023
@author: xgcerfo
"""


###############################################################################
def _plot_node_ids(G, ax, node_list=[], alpha=0.8, fontsize=8,
                   plot_node=False, node_size=15,
                   node_color='orange'):

    '''
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    '''

    Gnodes = set(G.nodes())

    if len(node_list) == 0:
        nodes = G.nodes()
    else:
        nodes = node_list
    for n in nodes:
        if n not in Gnodes:
            continue
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        if plot_node:
            ax.scatter(x, y, s=node_size, color=node_color)
        ax.annotate(str(n), xy=(x, y), alpha=alpha, fontsize=fontsize)

    return ax


###############################################################################
def plot_node_ids(G, ax, node_list=[], alpha=0.8, fontsize=8,
                  plot_node=False, node_size=15,
                  node_color='orange'):

    '''
    for label, x, y in zip(labels, data[:, 0], data[:, 1]):
    plt.annotate(
        label,
        xy=(x, y), xytext=(-20, 20),
        textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
    '''

    Gnodes = set(G.nodes())

    if len(node_list) == 0:
        nodes = G.nodes()
    else:
        nodes = node_list
    for n in nodes:
        if n not in Gnodes:
            continue
        x, y = G.nodes[n]['x'], G.nodes[n]['y']
        if plot_node:
            ax.scatter(x, y, s=node_size, color=node_color)
        ax.annotate(str(n), xy=(x, y), alpha=alpha, fontsize=fontsize)

    return ax
