from osgeo import gdal
import rasterio
import pandas
import geopandas
import copy
import numpy as np
import scipy.spatial
import networkx as nx
from rtree import index
import cv2
from shapely.geometry import Point, LineString
import utm

from plot_road import plot_graph
from matplotlib import pyplot as plt
from tqdm import tqdm

################################################################################
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

    # iterate through coorda and check if interpolated point has been passed
    # already or not
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

    # if we've reached here then that means we've encountered a self-loop and
    # the interpolated point is between the final midpoint and the the original
    # node
    i = len(coords) - 1
    cp = line.interpolate(distance)
    return [
        LineString(coords[:i] + [(cp.x, cp.y)]),
        LineString([(cp.x, cp.y)] + coords[i:])]

################################################################################
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

    # get distances from point to lines
    dist_list = []
    edge_list = []
    geom_list = []
    p = point  # Point(point_coords)
    for i, (u, v, key, data) in enumerate(G_.edges(keys=True, data=True)):
        # skip if u,v not in nearby nodes
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
        # print(p.distance(line))
        edge_list.append([u, v, key])
    # get closest edge
    min_idx = np.argmin(dist_list)
    min_dist = dist_list[min_idx]
    best_edge = edge_list[min_idx]
    best_geom = geom_list[min_idx]

    return best_edge, min_dist, best_geom

################################################################################
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

    TODO : Implement a version without renaming that tracks which node is
        closest to the desired point.

    Arguments
    ---------
    G_ : networkx graph
        Input networkx graph, with edges assumed to have a dictioary of
        properties that includes the 'geometry' key.
    point : shapely Point
        Shapely point containing (x, y) coordinates
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
    
    # find best edge and geom
    best_edge, min_dist, best_geom = get_closest_edge_from_G(
            G_, point, nearby_nodes_set=nearby_nodes_set,
            verbose=super_verbose)
    [u, v, key] = best_edge
    G_node_set = set(G_.nodes())

    if min_dist > max_distance_meters:
        return G_, {}, -1, -1

    else:
        # updated graph

        # skip if node exists already
        if node_id in G_node_set:
            return G_, {}, -1, -1

        line_geom = best_geom

        # Now combine with interpolated point on line
        new_point = line_geom.interpolate(line_geom.project(point))
        x, y = new_point.x, new_point.y
        
        # Length along line that is closest to the point
        line_proj = line_geom.project(new_point)
        
        # get others references coordinates
        y_pix, x_pix = rasterio.transform.rowcol(raster_file.transform, x, y)
        x_pix, y_pix = int(x_pix), int(-y_pix)
        df = pandas.DataFrame({'x':[x], 'y':[y]})
        gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.x, df.y), crs=raster_file.crs).to_crs(4326)
        lat, lon = gdf.geometry.x[0], gdf.geometry.y[0]

        #################
        # create new node

        # set properties
        node_props = {
            'osmid':   node_id,
            'lat':     lat,
            'lon':     lon,
            'x':       x,
            'y':       y,
            'x_pix':   x_pix,
            'y_pix':   y_pix,
            }

        # add node
        G_.add_node(node_id, **node_props)

        # assign, then update edge props for new edge
        _, _, edge_props_new = copy.deepcopy(
            list(G_.edges([u, v], data=True))[0])

        # cut line
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
            # get coincident node
            outnode = ''
            outnode_x, outnode_y = -1, -1
            x_p, y_p = new_point.x, new_point.y
            x_u, y_u = G_.nodes[u]['x'], G_.nodes[u]['y']
            x_v, y_v = G_.nodes[v]['x'], G_.nodes[v]['y']

            # sometimes it seems that the nodes aren't perfectly coincident,
            # so see if it's within a buffer
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

            # if the line cannot be split, that means that the new node
            # is coincident with an existing node.  Relabel, if desired
            if allow_renaming:
                node_props = G_.nodes[outnode]
                # A dictionary with the old labels as keys and new labels
                #  as values. A partial mapping is allowed.
                mapping = {outnode: node_id}
                Gout = nx.relabel_nodes(G_, mapping)
                return Gout, node_props, x_p, y_p

            else:
                # new node is already added, presumably at the exact location
                # of an existing node.  So just remove the best edge and make
                # an edge from new node to existing node, length should be 0.0

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

                # add edge of length 0 from new node to neareest existing node
                G_.add_edge(node_id, outnode, **edge_props_line1)
                return G_, node_props, x, y

        else:
            # else, create new edges
            line1, line2 = split_line

            # get distances
            u_loc = [G_.nodes[u]['x'], G_.nodes[u]['y']]
            v_loc = [G_.nodes[v]['x'], G_.nodes[v]['y']]
            # compare to first point in linestring
            geom_p0 = list(line_geom.coords)[0]
            dist_to_u = scipy.spatial.distance.euclidean(u_loc, geom_p0)
            dist_to_v = scipy.spatial.distance.euclidean(v_loc, geom_p0)
            # reverse edge order if v closer than u
            if dist_to_v < dist_to_u:
                line2, line1 = split_line
            
            # get geometry in pixels reference
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

            # add new edges
            edge_props_line1 = edge_props_new.copy()
            edge_props_line1['length'] = line1.length
            edge_props_line1['geometry'] = line1
            edge_props_line1['geometry_pix'] = line1_pix
            
            # line2
            edge_props_line2 = edge_props_new.copy()
            edge_props_line2['length'] = line2.length
            edge_props_line2['geometry'] = line2
            edge_props_line2['geometry_pix'] = line2_pix

            # check which direction linestring is travelling (it may be going
            # from v -> u, which means we need to reverse the linestring)
            # otherwise new edge is tangled
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

            # remove initial edge
            G_.remove_edge(u, v, key)

            return G_, node_props, x, y

################################################################################
def create_graph_midpoints(G_, raster_file, linestring_delta=50, is_curved_eps=0.03,
                           n_id_add_val=1, allow_renaming=True,
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

    # midpoints
    Gout = G_.copy()
    midpoint_name_val, midpoint_name_inc = np.max(G_.nodes())+n_id_add_val, 1
    # for u, v, data in tqdm(G_.edges(data=True)):
    for u, v, data in G_.edges(data=True):
        # curved line
        if 'geometry' in data:
            # first edge props and  get utm zone and letter
            edge_props_init = G_.edges([u, v])
            linelen = data['length']
            line = data['geometry']

            # ignore empty line or short line
            if linelen < 0.75*linestring_delta:
                continue

            # interpolate midpoints
            # if edge is short, use midpoint, else get evenly spaced points
            if linelen <= linestring_delta:
                interp_dists = [0.5 * line.length]
            else:
                # get evenly spaced points
                npoints = len(np.arange(0, linelen, linestring_delta)) + 1
                interp_dists = np.linspace(0, linelen, npoints)[1:-1]

            # create nodes
            for j, d in enumerate(interp_dists):

                midPoint = line.interpolate(d)
                xm0, ym0 = midPoint.xy
                point = Point(xm0[-1], ym0[-1])

                # add node to graph, with properties of u
                node_id = midpoint_name_val
                midpoint_name_val += midpoint_name_inc

                # add to graph
                Gout, node_props, _, _ = insert_point_into_G(
                    Gout, point, raster_file, node_id=node_id,
                    allow_renaming=allow_renaming,
                    verbose=super_verbose)

    return Gout

################################################################################
def distance(A,B):
	a = A[0]-B[0]
	b = A[1]-B[1]
	return np.sqrt(a*a + b*b)

def vNorm(v1):
	l = distance(v1,(0,0))+0.0000001
	return (v1[0]/l, v1[1]/l)

def anglediff(v1, v2):
	v1 = vNorm(v1)
	v2 = vNorm(v2)
	return v1[0]*v2[0] + v1[1] * v2[1]

def detect_local_minima(arr, mask, threshold = 0.5):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = scipy.ndimage.generate_binary_structure(len(arr.shape),2)
    # apply the local minimum filter; all locations of minimum value 
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (scipy.ndimage.minimum_filter(arr, footprint=neighborhood)==arr)
    # local_min is a mask that contains the peaks we are 
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    # 
    # we create the mask of the background
    background = (arr==0)
    # 
    # a little technicality: we must erode the background in order to 
    # successfully subtract it from local_min, otherwise a line will 
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = scipy.ndimage.binary_erosion(
        background, structure=neighborhood, border_value=1)
    # 
    # we obtain the final mask, containing only peaks, 
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where((detected_minima & (mask > threshold)))  

def decode_graph_encoding(imagegraph, max_degree=6, vector_norm=25, keypoints_thr=0.5, edges_thr=0.5, kp_limit=10000, snap_dist=100., angledistance_weight=100.):
    
    # Initialize decoding array
    decoding = np.zeros(imagegraph.shape[:-1], dtype=np.uint8)
    
    # Step-1: Find vertices
    # Step-1 (a): Find vertices through local minima detection. 
    vertexness = imagegraph[:,:,0]
    kp = np.copy(vertexness)
    smooth_kp = scipy.ndimage.gaussian_filter(np.copy(kp), 1)
    smooth_kp = smooth_kp / max(np.amax(smooth_kp),0.001)
    keypoints = detect_local_minima(-smooth_kp, smooth_kp, keypoints_thr)
    cc = 0
    
    # Step-1 (b): There could be a case where the local minima detection algorithm fails
    # to detect some of the vertices. 
    # For example, we have links a<-->b and b<-->c but b is missing. 
    # In this case, we use the edges a-->b and b<--c to recover b.
    # 
    # To do so, we locate the endpoint of each edge (from the detected vertices so far.),
    # draw all the endpoints on a numpy array (a 2D image), blur it, and use the same minima
    # detection algorithm to find vertices.
    edgeEndpointMap = np.zeros(imagegraph.shape[:-1])
    for i in range(len(keypoints[0])):
        if cc > kp_limit:
            break 
        
        cc += 1
        x,y = keypoints[0][i], keypoints[1][i]
        for j in range(max_degree):
            if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > keypoints_thr * keypoints_thr: # or thr < 0.2:
                x1 = int(x + vector_norm * imagegraph[x,y,2+4*j+2])
                y1 = int(y + vector_norm * imagegraph[x,y,2+4*j+3])
                if x1 >= 0 and x1 < imagegraph.shape[0] and y1 >= 0 and y1 < imagegraph.shape[1]:
                    edgeEndpointMap[x1,y1] = imagegraph[x,y,2+4*j] * imagegraph[x,y,0]

    edgeEndpointMap = scipy.ndimage.gaussian_filter(edgeEndpointMap, 3)
    edgeEndpoints = detect_local_minima(-edgeEndpointMap, edgeEndpointMap, keypoints_thr*keypoints_thr*keypoints_thr)
    
    # Step-1 (c): Create rtree index to speed up the queries.
    # We need to insert the vertices detected in Step-1(a) and Step-1(b) to the rtree.
    # For the vertices detected in Step-1(b), to avoid duplicated vertices, we only 
    # insert them when there are no nearby vertices around them. 
    idx = index.Index()
    cc = 0
    
    # Insert keypoints to the rtree
    for i in range(len(keypoints[0])):
        if cc > kp_limit:
            break 
        x,y = keypoints[0][i], keypoints[1][i]
        idx.insert(i,(x-1,y-1,x+1,y+1))
        cc += 1

    # Insert edge endpoints (the other vertex of the edge) to the rtree
    # To avoid duplicated vertices, we only insert the vertex when there is no
    # other vertex nearby.
    for i in range(len(edgeEndpoints[0])):
        if cc > kp_limit*2:
            break 
        x,y = edgeEndpoints[0][i], edgeEndpoints[1][i]
        candidates = list(idx.intersection((x-5,y-5,x+5,y+5)))
        if len(candidates) == 0:
            idx.insert(i + len(keypoints[0]),(x-1,y-1,x+1,y+1))
        cc += 1
    
    # Step-2 Connect the vertices to build a graph. 

    # endpoint lookup 
    neighbors = {}
    cc = 0
    for i in range(len(keypoints[0])):

        if cc > kp_limit:
            break 

        x,y = keypoints[0][i], keypoints[1][i]

        for j in range(max_degree):
            # imagegraph[x,y,2+4*j] --> edgeness
            # imagegraph[x,y,0] --> vertexness
            best_candidate = -1 
            if imagegraph[x,y,2+4*j] * imagegraph[x,y,0] > keypoints_thr*edges_thr and imagegraph[x,y,2+4*j] > edges_thr: 
                
                x1 = int(x + vector_norm * imagegraph[x,y,2+4*j+2])
                y1 = int(y + vector_norm * imagegraph[x,y,2+4*j+3])
                skip = False
                l = vector_norm * np.sqrt(imagegraph[x,y,2+4*j+2]*imagegraph[x,y,2+4*j+2] + imagegraph[x,y,2+4*j+3]*imagegraph[x,y,2+4*j+3])

                # We look for a candidate vertex to connect through three passes
                # Here, we use d(a-->b) to represent the distance metric for edge a-->b .
                # Pass-1 For a link a<-->b, we connect them only if d(a-->b) + d(a<--b) <= snap_dist.
                # Pass-2 (relaxed) For a link a<-->b, we connect them only if 2*d(a-->b) <= snap_dist or 2*d(a<--b) <= snap_dist.
                # Pass-3 (more relaxed) For a link a<-->b, we connect them only if d(a-->b) <= snap_dist or d(a<--b) <= snap_dist.
                # 
                # In Pass-1 and Pass-2, we only consider the keypoints detected directly by the minima detection algorithm (Step-1(a)).
                # In Pass-3, we only consider the edge end points detected in Step-1(b)
                best_candidate = -1 
                min_distance = snap_dist #15.0 

                candidates = list(idx.intersection((x1-20,y1-20,x1+20,y1+20)))
                
                # Pass-1 (restrict distance metric)
                for candidate in candidates:
                    # only snap to keypoints 
                    if candidate >= len(keypoints[0]):
                        continue

                    if candidate < len(keypoints[0]):
                        x_c = keypoints[0][candidate]
                        y_c = keypoints[1][candidate]
                    else:
                        x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
                        y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

                    d = distance((x_c,y_c), (x1,y1))
                    if d > l :
                        continue 

                    # vector from the edge endpoint (the other side of the edge) to the current vertex. 
                    v0 = (x - x_c, y - y_c)

                    min_sd = angledistance_weight

                    for jj in range(max_degree):
                        if imagegraph[x_c,y_c,2+4*jj] * imagegraph[x_c,y_c,0] > keypoints_thr*edges_thr and imagegraph[x,y,2+4*jj] > edges_thr:
                            vc = (vector_norm * imagegraph[x_c,y_c,2+4*jj+2], vector_norm * imagegraph[x_c,y_c,2+4*jj+3])

                            # cosine distance
                            ad = 1.0 - anglediff(v0,vc)
                            ad = ad * angledistance_weight 

                            if ad < min_sd:
                                min_sd = ad 

                    d = d + min_sd


                    # cosine distance between the original output edge direction and the edge direction after snapping.
                    v1 = (x_c - x, y_c - y)
                    v2 = (x1 - x, y1 - y)
                    # cosine distance 
                    ad = 1.0 - anglediff(v1,v2) # -1 to 1
                    d = d + ad * angledistance_weight # 0.15 --> 15 degrees 

                    if d < min_distance:
                        min_distance = d 
                        best_candidate = candidate

                # Pass-2 (relax the distance metric)
                min_distance = snap_dist #15.0 
                # only need the second pass when there is no good candidate found in the first pass. 
                if best_candidate == -1:
                    for candidate in candidates:
                        # only snap to keypoints 
                        if candidate >= len(keypoints[0]):
                            continue

                        if candidate < len(keypoints[0]):
                            x_c = keypoints[0][candidate]
                            y_c = keypoints[1][candidate]
                        else:
                            x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
                            y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

                        d = distance((x_c,y_c), (x1,y1))
                        if d > l*0.5 :
                            continue 

                        # cosine distance between the original output edge direction and the edge direction after snapping.  
                        v1 = (x_c - x, y_c - y)
                        v2 = (x1 - x, y1 - y)

                        ad = 1.0 - anglediff(v1,v2) # -1 to 1
                        d = d + ad * angledistance_weight * 2 # 0.15 --> 30

                        if d < min_distance:
                            min_distance = d 
                            best_candidate = candidate

                # Pass-3 (relax the distance metric even more)
                if best_candidate == -1:
                    for candidate in candidates:
                        # only snap to edge endpoints 
                        if candidate < len(keypoints[0]):
                            continue

                        if candidate < len(keypoints[0]):
                            x_c = keypoints[0][candidate]
                            y_c = keypoints[1][candidate]
                        else:
                            x_c = edgeEndpoints[0][candidate-len(keypoints[0])]
                            y_c = edgeEndpoints[1][candidate-len(keypoints[0])]

                        d = distance((x_c,y_c), (x1,y1))
                        if d > l :
                            continue 

                        v1 = (x_c - x, y_c - y)
                        v2 = (x1 - x, y1 - y)

                        ad = 1.0 - anglediff(v1,v2) # -1 to 1
                        d = d + ad * angledistance_weight # 0.15 --> 15

                        if d < min_distance:
                            min_distance = d 
                            best_candidate = candidate

                if best_candidate != -1 :
                    if best_candidate < len(keypoints[0]):
                        x1 = keypoints[0][best_candidate]
                        y1 = keypoints[1][best_candidate]
                    else:
                        x1 = edgeEndpoints[0][best_candidate-len(keypoints[0])]
                        y1 = edgeEndpoints[1][best_candidate-len(keypoints[0])]
                else:
                    skip = True

            # visualization : draw the edges and add them in 'neighbors'
            if best_candidate != -1:# or True:
                nk1 = (x1,y1)
                nk2 = (x,y)

                if nk1 != nk2:
                    if nk1 in neighbors:
                        if nk2 in neighbors[nk1]:
                            pass
                        else:
                            neighbors[nk1].append(nk2)
                    else:
                        neighbors[nk1] = [nk2]

                    if  nk2 in neighbors:
                        if nk1 in neighbors[nk2]:
                            pass 
                        else:
                            neighbors[nk2].append(nk1)
                    else:
                        neighbors[nk2] = [nk1]

                cv2.line(decoding,(y1,x1), (y,x), 255, 2)
        cc += 1
        
    return decoding