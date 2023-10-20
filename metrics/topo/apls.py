# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 12:32:19 2017
@author: avanetten

Updated on Mon Sep 11 09:00:00 2023
@author: xgcerfo
"""

import networkx as nx
import scipy.spatial
import scipy.stats
import numpy as np
import copy
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import os
import sys
import pandas as pd
import geopandas as gpd
import rasterio

path_apls_src = os.path.dirname(os.path.realpath(__file__))
path_apls = os.path.dirname(path_apls_src)

sys.path.append(path_apls_src)


###############################################################################
def create_edge_linestrings(G_, remove_redundant=True, verbose=False):

    """
    Ensure all edges have the 'geometry' tag, use shapely linestrings.

    Notes
    -----
    If identical edges exist, remove extras.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that may or may not include 'geometry'.
    remove_redundant : boolean
        Switch to remove identical edges, if they exist.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    G_ : networkx graph
        Updated graph with every edge containing the 'geometry' tag.
    """

    # Clean out redundant edges with identical geometry
    edge_seen_set = set([])
    geom_seen = []
    bad_edges = []

    for i, (u, v, data) in enumerate(G_.edges(data=True)):
        # Create linestring if no geometry reported
        if 'geometry' not in data:
            sourcex, sourcey = G_.nodes[u]['x'],  G_.nodes[u]['y']
            targetx, targety = G_.nodes[v]['x'],  G_.nodes[v]['y']
            line_geom = LineString([Point(sourcex, sourcey),
                                    Point(targetx, targety)])
            data['geometry'] = line_geom

            # Get reversed line
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
        else:
            # Check which direction linestring is travelling (it may be going
            #   from v -> u, which means we need to reverse the linestring)
            #   otherwise new edge is tangled
            line_geom = data['geometry']
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            coords = list(data['geometry'].coords)[::-1]
            line_geom_rev = LineString(coords)
            if dist_to_u > dist_to_v:
                data['geometry'] = line_geom_rev

        # Flag redundant edges
        if remove_redundant:
            if i == 0:
                edge_seen_set = set([(u, v)])
                edge_seen_set.add((v, u))
                geom_seen.append(line_geom)

            else:
                if ((u, v) in edge_seen_set) or ((v, u) in edge_seen_set):
                    # Test if geoms have already been seen
                    for geom_seen_tmp in geom_seen:
                        if (line_geom == geom_seen_tmp) \
                                or (line_geom_rev == geom_seen_tmp):
                            bad_edges.append((u, v))
                            if verbose:
                                print("\nRedundant edge:", u, v)
                else:
                    edge_seen_set.add((u, v))
                    geom_seen.append(line_geom)
                    geom_seen.append(line_geom_rev)

    if remove_redundant:
        if verbose:
            print("\nedge_seen_set:", edge_seen_set)
            print("redundant edges:", bad_edges)
        for (u, v) in bad_edges:
            if G_.has_edge(u, v):
                G_.remove_edge(u, v)

    return G_


###############################################################################
def cut_linestring(line, distance, verbose=False):

    """
    Cuts a shapely linestring at a specified distance from its starting point.

    Notes
    ----
    Return orignal linestring if distance <= 0 or greater than the length of
    the line.
    Reference:
        http://toblerity.org/shapely/manual.html#linear-referencing-methods

    Arguments
    ---------
    line : shapely linestring
        Input shapely linestring to cut.
    distanct : float
        Distance from start of line to cut it in two.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    [line1, line2] : list
        Cut linestrings.  If distance <= 0 or greater than the length of
        the line, return input line.
    """

    if verbose:
        print("Cutting linestring at distance", distance, "...")
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    # Iterate through coorda and check if interpolated point has been passed already or not
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pdl = line.project(Point(p))
        if verbose:
            print(i, p, "line.project point:", pdl)
        if pdl == distance:
            return [
                LineString(coords[:i+1]),
                LineString(coords[i:])]
        if pdl > distance:
            cp = line.interpolate(distance)
            return [
                LineString(coords[:i] + [(cp.x, cp.y)]),
                LineString([(cp.x, cp.y)] + coords[i:])]

    # If we've reached here then that means we've encountered a self-loop and
    #  the interpolated point is between the final midpoint and the the original
    #  node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]


###############################################################################
def get_closest_edge_from_G(G_, point, nearby_nodes_set=set([]),
                            verbose=False):

    """
    Return closest edge to point, and distance to said edge.

    Notes
    -----
    Just discovered a similar function:
        https://github.com/gboeing/osmnx/blob/master/osmnx/utils.py#L501

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.

    Returns
    -------
    best_edge, min_dist, best_geom : tuple
        best_edge is the closest edge to the point
        min_dist is the distance to that edge
        best_geom is the geometry of the ege
    """

    # Get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # Skip if u,v not in nearby nodes
        if len(nearby_nodes_set) > 0:
            if (u not in nearby_nodes_set) and (v not in nearby_nodes_set):
                continue
        if verbose:
            print(("u,v,key,data:", u, v, key, data))
            print(("  type data['geometry']:", type(data['geometry'])))
        try:
            line = data['geometry']
        except KeyError:
            line = data['attr_dict']['geometry']
        geom_list.append(line)
        dist_list.append(p.distance(line))
        edge_list.append([u, v, key])
    # Get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom


###############################################################################
def insert_point_into_G(G_, point, raster_file, node_id=100000, max_distance_meters=5,
                        nearby_nodes_set=set([]), allow_renaming=True,
                        verbose=False, super_verbose=False):

    """
    Insert a new node in the graph closest to the given point.

    Notes
    -----
    If the point is too far from the graph, don't insert a node.
    Assume all edges have a linestring geometry
    http://toblerity.org/shapely/manual.html#object.simplify
    Sometimes the point to insert will have the same coordinates as an
    existing point.  If allow_renaming == True, relabel the existing node.
    convert linestring to multipoint?
     https://github.com/Toblerity/Shapely/issues/190

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
    raster_file : Python file object
        Origin mask
    node_id : int
        Unique identifier of node to insert. Defaults to ``100000``.
    max_distance_meters : float
        Maximum distance in meters between point and graph. Defaults to ``5``.
    nearby_nodes_set : set
        Set of possible edge endpoints to search.  If nearby_nodes_set is not
        empty, only edges with a node in this set will be checked (this can
        greatly speed compuation on large graphs).  If nearby_nodes_set is
        empty, check all possible edges in the graph.
        Defaults to ``set([])``.
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    G_, node_props, min_dist : tuple
        G_ is the updated graph
        node_props gives the properties of the inserted node
        min_dist is the distance from the point to the graph
    """

    # Find best edge and geom
    best_edge, min_dist, best_geom = get_closest_edge_from_G(
        G_, point, nearby_nodes_set=nearby_nodes_set,
        verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if min_dist > max_distance_meters:
        return G_, {}, -1, -1

    else:
        # Updated graph
        # Skip if node exists already
        if node_id in G_node_set:
            return G_, {}, -1, -1

        line_geom = best_geom

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y

        # Length along line that is closest to the point
        line_proj = line_geom.project(new_point)

        # Get others references coordinates
        y_pix, x_pix = rasterio.transform.rowcol(raster_file.transform, x, y)
        x_pix, y_pix = int(x_pix), int(-y_pix)
        df = pd.DataFrame({'x': [x], 'y': [y]})
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs=raster_file.crs).to_crs(
            4326)
        lat, lon = gdf.geometry.x[0], gdf.geometry.y[0]

        #################
        # Create new node
        # Set properties
        node_props = {
            'osmid': node_id,
            'lat': lat,
            'lon': lon,
            'x': x,
            'y': y,
            'x_pix': x_pix,
            'y_pix': y_pix,
        }

        # Add node
        G_.add_node(node_id, **node_props)

        # Assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])

        # Cut line
        split_line = cut_linestring(line_geom, line_proj)
        if split_line is None:
            print("Failure in cut_linestring()...")
            print("type(split_line):", type(split_line))
            print("split_line:", split_line)
            print("line_geom:", line_geom)
            print("line_geom.length:", line_geom.length)
            print("line_proj:", line_proj)
            print("min_dist:", min_dist)
            return G_, {}, 0, 0

        if len(split_line) == 1:
            # Get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']

            # Sometimes it seems that the nodes aren't perfectly coincident, so see if it's within a buffer
            buff = 0.05  # meters
            if (abs(x_p - x_u) <= buff) and (abs(y_p - y_u) <= buff):
                outnode = u
                outnode_x, outnode_y = x_u, y_u
            elif (abs(x_p - x_v) <= buff) and (abs(y_p - y_v) <= buff):
                outnode = v
                outnode_x, outnode_y = x_v, y_v
            else:
                print("Error in determining node coincident with node: "
                      + str(node_id) + " along edge: " + str(best_edge))
                print("x_p, y_p:", x_p, y_p)
                print("x_u, y_u:", x_u, y_u)
                print("x_v, y_v:", x_v, y_v)
                # return
                return G_, {}, 0, 0

            # If the line cannot be split, that means that the new node
            #  is coincident with an existing node. Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                return Gout, node_props, x_p, y_p

            else:
                # New node is already added, presumably at the exact location
                #  of an existing node.  So just remove the best edge and make
                #  an edge from new node to existing node, length should be 0.0

                line1 = LineString([new_point, Point(outnode_x, outnode_y)])
                line1_pix = []
                for p in line1.coords:
                    y_pix, x_pix = rasterio.transform.rowcol(raster_file.transform, p[0], p[1])
                    y_pix = -y_pix
                    line1_pix.append(Point(x_pix, y_pix))
                line1_pix = LineString(line1_pix)
                edge_props_line1 = edge_props_new.copy()
                edge_props_line1['length'] = line1.length
                edge_props_line1['geometry'] = line1
                edge_props_line1['geometry_pix'] = line1_pix
                edge_props_line1['start'] = node_id
                edge_props_line1['end'] = outnode
                # make sure length is zero
                if line1.length > buff:
                    print("Nodes should be coincident and length 0!")
                    print("  line1.length:", line1.length)
                    print("  x_u, y_u :", x_u, y_u)
                    print("  x_v, y_v :", x_v, y_v)
                    print("  x_p, y_p :", x_p, y_p)
                    print("  new_point:", new_point)
                    print("  Point(outnode_x, outnode_y):",
                          Point(outnode_x, outnode_y))
                    return

                # Add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

        # Else, create new edges
        else:
            line1, line2 = split_line

            # Get distances
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # Compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # Reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line

            # Get geometry in pixels reference
            line1_pix, line2_pix = [], []
            for p in line1.coords:
                y_pix, x_pix = rasterio.transform.rowcol(raster_file.transform, p[0], p[1])
                y_pix = -y_pix
                line1_pix.append(Point(x_pix, y_pix))
            for p in line2.coords:
                y_pix, x_pix = rasterio.transform.rowcol(raster_file.transform, p[0], p[1])
                y_pix = -y_pix
                line2_pix.append(Point(x_pix, y_pix))
            line1_pix = LineString(line1_pix)
            line2_pix = LineString(line2_pix)

            # Add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            edge_props_line1['geometry_pix'] = line1_pix

            # Line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            edge_props_line2['geometry_pix'] = line2_pix

            # Check which direction linestring is travelling (it may be going
            #  from v -> u, which means we need to reverse the linestring)
            #  otherwise new edge is tangled
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            if dist_to_u < dist_to_v:
                edge_props_line1['start'] = u
                edge_props_line1['end'] = node_id
                G_.add_edge(u, node_id, **edge_props_line1)
                edge_props_line2['start'] = node_id
                edge_props_line2['end'] = v
                G_.add_edge(node_id, v, **edge_props_line2)
            else:
                edge_props_line1['start'] = node_id
                edge_props_line1['end'] = u
                G_.add_edge(node_id, u, **edge_props_line1)
                edge_props_line2['start'] = v
                edge_props_line2['end'] = node_id
                G_.add_edge(v, node_id, **edge_props_line2)

            # Remove initial edge
            G_.remove_edge(u, v, key)

            return G_, node_props, x, y


###############################################################################
def create_graph_midpoints(G_, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=True,
                           figsize=(0, 0),
                           verbose=False, super_verbose=False):

    """
    Insert midpoint nodes into long edges on the graph.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    linestring_delta : float
        Distance in meters between linestring midpoints. Defaults to ``50``.
    is_curved_eps : float
        Minumum curvature for injecting nodes (if curvature is less than this
        value, no midpoints will be injected). If < 0, always inject points
        on line, regardless of curvature.  Defaults to ``0.3``.
    n_id_add_val : int
        Sets min midpoint id above existing nodes
        e.g.: G.nodes() = [1,2,4], if n_id_add_val = 5, midpoints will
        be [9,10,11,...]
    allow_renameing : boolean
        Switch to allow renaming of an existing node with node_id if the
        existing node is closest to the point. Defaults to ``False``.
    figsize : tuple
        Figure size for optional plot. Defaults to ``(0,0)`` (no plot).
    verbose : boolean
        Switch to print relevant values to screen.  Defaults to ``False``.
    super_verbose : boolean
        Switch to print mucho values to screen.  Defaults to ``False``.

    Returns
    -------
    Gout, xms, yms : tuple
        Gout is the updated graph
        xms, yms are coordinates of the inserted points
    """

    if len(G_.nodes()) == 0:
        return G_, [], []

    # Midpoints
    xms, yms = [], []
    Gout = G_.copy()
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    for u, v, data in G_.edges(data=True):

        # Curved line
        if 'geometry' in data:

            # First edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])

            linelen = data['length']
            line = data['geometry']

            xs, ys = line.xy  # For plotting

            # Check if curved or not
            #################
            minx, miny, maxx, maxy = line.bounds
            # Get euclidean distance
            dst = scipy.spatial.distance.euclidean([minx, miny], [maxx, maxy])
            # Ignore if almost straight
            if np.abs(dst - linelen) / linelen < is_curved_eps:
                continue
            #################

            # Also ignore super short lines
            #################
            if linelen < 0.75*linestring_delta:
                continue
            #################

            if verbose:
                print("create_graph_midpoints()...")
                print("  u,v:", u, v)
                print("  data:", data)
                print("  edge_props_init:", edge_props_init)

            # Interpolate midpoints
            # If edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # Get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]
                if verbose:
                    print("  interp_dists:", interp_dists)

            # Create nodes
            node_id_new_list = []
            xms_tmp, yms_tmp = [], []
            for j, d in enumerate(interp_dists):
                if verbose:
                    print("    ", j, "interp_dist:", d)

                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                xm = xm0[-1]
                ym = ym0[-1]
                point = Point(xm, ym)
                xms.append(xm)
                yms.append(ym)
                xms_tmp.append(xm)
                yms_tmp.append(ym)
                if verbose:
                    print("    midpoint:", xm, ym)

                # Add node to graph, with properties of u
                node_id = midpoint_name_val
                midpoint_name_val += midpoint_name_inc
                node_id_new_list.append(node_id)
                if verbose:
                    print("    node_id:", node_id)

                # Add to graph
                Gout, node_props, _, _ = insert_point_into_G(
                    Gout, point, node_id=node_id,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

        # Plot, if desired
        if figsize != (0, 0):
            fig, (ax) = plt.subplots(1, 1, figsize=(1*figsize[0], figsize[1]))
            ax.plot(xs, ys, color='#6699cc', alpha=0.7,
                    linewidth=3, solid_capstyle='round', zorder=2)
            ax.scatter(xm, ym, color='red')
            ax.set_title('Line Midpoint')
            plt.axis('equal')

    return Gout, xms, yms


###############################################################################
def _clean_sub_graphs(G_, min_length=80, max_nodes_to_skip=100,
                      weight='length', verbose=True,
                      super_verbose=False):

    """
    Remove subgraphs with a max path length less than min_length,
    if the subgraph has more than max_noxes_to_skip, don't check length
       (this step great reduces processing time)
    """

    if len(G_.nodes()) == 0:
        return G_

    if verbose:
        print("Running clean_sub_graphs...")
    sub_graphs = list(nx.connected_component_subgraphs(G_))
    bad_nodes = []
    if verbose:
        print(" len(G_.nodes()):", len(G_.nodes()))
        print(" len(G_.edges()):", len(G_.edges()))
    if super_verbose:
        print("G_.nodes:", G_.nodes())
        edge_tmp = G_.edges()[np.random.randint(len(G_.edges()))]
        print(edge_tmp, "G.edge props:", G_.edges[edge_tmp[0]][edge_tmp[1]])

    for G_sub in sub_graphs:
        # Don't check length if too many nodes in subgraph
        if len(G_sub.nodes()) > max_nodes_to_skip:
            continue

        else:
            all_lengths = dict(
                nx.all_pairs_dijkstra_path_length(G_sub, weight=weight))
            if super_verbose:
                print("  \nGs.nodes:", G_sub.nodes())
                print("  all_lengths:", all_lengths)
            # Get all lenghts
            lens = []

            for u in all_lengths.keys():
                v = all_lengths[u]
                for uprime in v.keys():
                    vprime = v[uprime]
                    lens.append(vprime)
                    if super_verbose:
                        print("  u, v", u, v)
                        print("    uprime, vprime:", uprime, vprime)
            max_len = np.max(lens)
            if super_verbose:
                print("  Max length of path:", max_len)
            if max_len < min_length:
                bad_nodes.extend(G_sub.nodes())
                if super_verbose:
                    print(" appending to bad_nodes:", G_sub.nodes())

    # Remove bad_nodes
    G_.remove_nodes_from(bad_nodes)
    if verbose:
        print(" num bad_nodes:", len(bad_nodes))
        print(" len(G'.nodes()):", len(G_.nodes()))
        print(" len(G'.edges()):", len(G_.edges()))
    if super_verbose:
        print("  G_.nodes:", G_.nodes())

    return G_