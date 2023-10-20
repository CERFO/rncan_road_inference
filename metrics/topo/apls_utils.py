# -*- coding: utf-8 -*-
"""
Created on Mon May  6 14:05:30 2019
@author: avanetten

Updated on Mon Sep 11 09:00:00 2023
@author: xgcerfo
"""

import numpy as np
import scipy.spatial
import time
import os
import sys

# Add apls path and import apls_tools
path_apls_src = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_apls_src)


###############################################################################
def nodes_near_point(x, y, kdtree, kd_idx_dic, x_coord='x', y_coord='y',
                     n_neighbors=-1,
                     radius_m=150,
                     verbose=False):

    """
    Get nodes near the given point.

    Notes
    -----
    if n_neighbors < 0, query based on distance,
    else just return n nearest neighbors

    Arguments
    ---------
    x : float
        x coordinate of point
    y: float
        y coordinate of point
    kdtree : scipy.spatial.kdtree
        kdtree of nondes in graph
    kd_idx_dic : dict
        Dictionary mapping kdtree entry to node name
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.
    n_neighbors : int
        Neareast number of neighbors to return. If < 0, ignore.
        Defaults to ``-1``.
    radius_meters : float
        Radius to search for nearest neighbors
    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    point = [x, y]

    # Query kd tree for nodes of interest
    if n_neighbors > 0:
        node_names, idxs_refine, dists_m_refine = _query_kd_nearest(
            kdtree, kd_idx_dic, point, n_neighbors=n_neighbors)
    else:
        node_names, idxs_refine, dists_m_refine = _query_kd_ball(
            kdtree, kd_idx_dic, point, radius_m)

    if verbose:
        print(("subgraph node_names:", node_names))

    return node_names, dists_m_refine


###############################################################################
def _nodes_near_origin(G_, node, kdtree, kd_idx_dic,
                      x_coord='x', y_coord='y', radius_m=150, verbose=False):

    '''Get nodes a given radius from the desired node.  G_ should be the 
    maximally simplified graph'''

    # Get node coordinates
    n_props = G_.nodes[node]
    x0, y0 = n_props[x_coord], n_props[y_coord]
    point = [x0, y0]

    # Query kd tree for nodes of interest
    node_names, idxs_refine, dists_m_refine = _query_kd_ball(
        kdtree, kd_idx_dic, point, radius_m)
    if verbose:
        print(("subgraph node_names:", node_names))

    return node_names, dists_m_refine


###############################################################################
def G_to_kdtree(G_, x_coord='x', y_coord='y', verbose=False):

    """
    Create kd tree from node positions.

    Notes
    -----
    (x, y) = (lon, lat)
    kd_idx_dic maps kdtree entry to node name:
        kd_idx_dic[i] = n (n in G.nodes())
    x_coord can be in utm (meters), or longitude

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with nodes assumed to have a dictioary of
        properties that includes position
    x_coord : str
        Name of x_coordinate, can be 'x' or 'lon'. Defaults to ``'x'``.
    y_coord : str
        Name of y_coordinate, can be 'y' or 'lat'. Defaults to ``'y'``.

    Returns
    -------
    kd_idx_dic, kdtree, arr : tuple
        kd_idx_dic maps kdtree entry to node name
        kdree is the actual kdtree
        arr is the numpy array of node positions
    """

    nrows = len(G_.nodes())
    ncols = 2
    kd_idx_dic = {}
    arr = np.zeros((nrows, ncols))
    # Populate node array
    t1 = time.time()
    for i, n in enumerate(G_.nodes()):
        n_props = G_.nodes[n]
        if x_coord == 'lon':
            lat, lon = n_props['lat'], n_props['lon']
            x, y = lon, lat
        else:
            x, y = n_props[x_coord], n_props[y_coord]

        arr[i] = [x, y]
        kd_idx_dic[i] = n

    # Now create kdtree from numpy array
    kdtree = scipy.spatial.KDTree(arr)
    if verbose:
        print("Time to create k-d tree:", time.time() - t1, "seconds")
    return kd_idx_dic, kdtree, arr


###############################################################################
def _query_kd_nearest(kdtree, kd_idx_dic, point, n_neighbors=10,
                      distance_upper_bound=1000, keep_point=True):

    '''
    Query the kd-tree for neighbors
    Return nearest node names, distances, nearest node indexes
    If not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=n_neighbors,
                                 distance_upper_bound=distance_upper_bound)

    idxs_refine = list(np.asarray(idxs))
    dists_m_refine = list(dists_m)
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _query_kd_ball(kdtree, kd_idx_dic, point, r_meters, keep_point=True):

    '''
    Query the kd-tree for neighbors within a distance r of the point
    Return nearest node names, distances, nearest node indexes
    if not keep_point, remove the origin point from the list
    '''

    dists_m, idxs = kdtree.query(point, k=500, distance_upper_bound=r_meters)
    # Keep only points within distance and greaater than 0?
    if not keep_point:
        f0 = np.where((dists_m <= r_meters) & (dists_m > 0))
    else:
        f0 = np.where((dists_m <= r_meters))
    idxs_refine = list(np.asarray(idxs)[f0])
    dists_m_refine = list(dists_m[f0])
    node_names = [kd_idx_dic[i] for i in idxs_refine]

    return node_names, idxs_refine, dists_m_refine


###############################################################################
def _get_node_positions(G_, x_coord='x', y_coord='y'):

    '''Get position array for all nodes'''

    nrows = len(G_.nodes())
    ncols = 2
    arr = np.zeros((nrows, ncols))
    # Populate node array
    for i, n in enumerate(G_.nodes()):
        n_props = G_.nodes[n]
        x, y = n_props[x_coord], n_props[y_coord]
        arr[i] = [x, y]
    return arr
