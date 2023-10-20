import rasterio
from rasterio import features
import geopandas
import numpy as np
import scipy
import cv2
from rtree import index

###########################################################################################################################
# Global postprocess utils
#
###########################################################################################################################
def clip_to_min_extent(annotations_index_path, pred, pred_name, pred_crs, kwds, nodata_val):
    
    # Read index geodataframe
    index_gdf = geopandas.read_file(annotations_index_path)
    zones_names = index_gdf['nom_etendue'].values.tolist()

    # Get annotated zone
    idx = [zones_names.index(zone) for zone in zones_names if zone.startswith(pred_name.split('_')[0])]
    zone_gdf = index_gdf.iloc[idx].to_crs(pred_crs)
    shapes = ((geom,1) for geom in zone_gdf.geometry)
    zone = features.rasterize(shapes=shapes, fill=0, out_shape=pred.shape, dtype=np.uint8, transform=kwds['transform'])

    # Find minimum extent
    data_indexes = np.asarray(np.where(zone==1))
    i_min, i_max = data_indexes[0].min(), data_indexes[0].max()
    j_min, j_max = data_indexes[1].min(), data_indexes[1].max()

    # Set nodatas outside of annotated zone
    pred[zone==0] = nodata_val
    pred = pred[i_min:i_max, j_min:j_max]

    # Get new transformation
    res = kwds['transform'][0]
    west, _ = rasterio.transform.xy(kwds['transform'], i_max, j_min, offset='ll')
    _, north = rasterio.transform.xy(kwds['transform'], i_min, j_max, offset='ll')
    width = pred.shape[1]
    height = pred.shape[0]
    transform = rasterio.transform.Affine(res, 0.0, west, 0.0, -res, north)

    # Set new metadata
    kwds['count'] = 1
    kwds['dtype'] = np.uint8
    kwds['width'] = int(width)
    kwds['height'] = int(height)
    kwds['transform'] = transform
    kwds['nodata'] = nodata_val
    
    return pred, kwds

def calculate_idx(idx, last_bool, file_shape, overlap, load_size):
    if last_bool:
        done_bool = True
    else:
        done_bool = False
        idx += int(load_size * (1-overlap))
        if idx+load_size >= file_shape:
            idx = file_shape - load_size
            last_bool = True
    return idx, last_bool, done_bool

###########################################################################################################################
# GTE decoding utils
#
###########################################################################################################################
def prepare_gte(compact_gte, max_degree):
    
    # prediction normalization
    compact_gte = np.true_divide(compact_gte[:-1], compact_gte[-1], out=np.zeros_like(compact_gte[:-1]), where=compact_gte[-1]!=0)
    compact_gte = np.moveaxis(compact_gte, 0, -1)
    
    # gte encryption
    decoding = np.zeros([compact_gte.shape[0], compact_gte.shape[1], 28])
    decoding[...,0] = compact_gte[...,0]
    decoding[...,1] = 1 - compact_gte[...,0]
    for i in range(max_degree):
        decoding[..., 4*i + 2] = compact_gte[..., 3*i + 1]
        decoding[..., 4*i + 3] = 1. - compact_gte[..., 3*i + 1]
        decoding[..., 4*i + 4] = compact_gte[..., 3*i + 2]
        decoding[..., 4*i + 5] = compact_gte[..., 3*i + 3]
    return decoding

def distance(A,B):
    a = A[0]-B[0]
    b = A[1]-B[1]
    return np.sqrt(a*a + b*b)

def vNorm(v1):
    l = distance(v1,(0,0))+0.0000001
    return (v1[0]/l, v1[1]/l)

def anglediff(v1, v2):
    # Testing #######################################################################################################	v1 = vNorm(v1)
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

def decode_graph_encoding(imagegraph, max_degree=6, vector_norm=25, keypoints_thr=0.5, edges_thr=0.5, kp_limit=10000, candidate_search_dist=20., snap_dist=100., angledistance_weight=100., line_thickness=2):
    
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
                
                candidates = list(idx.intersection((x1-candidate_search_dist,y1-candidate_search_dist,x1+candidate_search_dist,y1+candidate_search_dist)))
                
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
            if best_candidate != -1:
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

                cv2.line(decoding,(y1,x1), (y,x), 255, line_thickness)
        cc += 1
        
    return decoding