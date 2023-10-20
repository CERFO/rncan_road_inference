import os
from glob import glob
import numpy as np
from osgeo import gdal
import rasterio
from rasterio import windows
from utils import clip_to_min_extent
import cv2
# from scipy.ndimage import gaussian_filter
# from scipy.ndimage import grey_closing
from scipy.spatial import distance
from skimage.morphology import skeletonize
from skimage.graph import route_through_array
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from matplotlib import pyplot as plt
import copy

def get_skeleton(seg):
    # smooth
    uniform_kernel = np.ones((3,3)).astype(np.uint8)
    ellipse_kernel_3x3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    seg = cv2.morphologyEx(seg, cv2.MORPH_CLOSE, uniform_kernel)
    seg = cv2.dilate(seg, ellipse_kernel_3x3, iterations=1)
    seg = cv2.erode(seg, ellipse_kernel_3x3, iterations=1)
    seg = cv2.blur(seg, (5,5))
    (T, threshInv) = cv2.threshold(seg, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # skeletonize
    skel = skeletonize(seg).astype(np.uint8)
    return skel

def find_neighbours(x,y,img):
    """Return 8-neighbours of image point P1(x,y), in a clockwise order"""
    #img = image
    x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
    return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

def skeleton_intersections(skeleton):
    '''https://stackoverflow.com/questions/41705405/finding-intersections-of-a-skeletonised-image-in-python-opencv'''
    """ Given a skeletonised image, it will give the coordinates of the intersections of the skeleton.

    Keyword arguments:
    skeleton -- the skeletonised image to detect the intersections of

    Returns: 
    List of 2-tuples (x,y) containing the intersection coordinates
    """
    # A biiiiiig list of valid intersections             2 3 4
    # These are in the format shown to the right         1 C 5
    #                                                    8 7 6 
    validIntersection = [[0,1,0,1,0,0,1,0],[0,0,1,0,1,0,0,1],[1,0,0,1,0,1,0,0],
                         [0,1,0,0,1,0,1,0],[0,0,1,0,0,1,0,1],[1,0,0,1,0,0,1,0],
                         [0,1,0,0,1,0,0,1],[1,0,1,0,0,1,0,0],[0,1,0,0,0,1,0,1],
                         [0,1,0,1,0,0,0,1],[0,1,0,1,0,1,0,0],[0,0,0,1,0,1,0,1],
                         [1,0,1,0,0,0,1,0],[1,0,1,0,1,0,0,0],[0,0,1,0,1,0,1,0],
                         [1,0,0,0,1,0,1,0],[1,0,0,1,1,1,0,0],[0,0,1,0,0,1,1,1],
                         [1,1,0,0,1,0,0,1],[0,1,1,1,0,0,1,0],[1,0,1,1,0,0,1,0],
                         [1,0,1,0,0,1,1,0],[1,0,1,1,0,1,1,0],[0,1,1,0,1,0,1,1],
                         [1,1,0,1,1,0,1,0],[1,1,0,0,1,0,1,0],[0,1,1,0,1,0,1,0],
                         [0,0,1,0,1,0,1,1],[1,0,0,1,1,0,1,0],[1,0,1,0,1,1,0,1],
                         [1,0,1,0,1,1,0,0],[1,0,1,0,1,0,0,1],[0,1,0,0,1,0,1,1],
                         [0,1,1,0,1,0,0,1],[1,1,0,1,0,0,1,0],[0,1,0,1,1,0,1,0],
                         [0,0,1,0,1,1,0,1],[1,0,1,0,0,1,0,1],[1,0,0,1,0,1,1,0],
                         [1,0,1,1,0,1,0,0]]
    image = skeleton.copy()
    intersections = list()
    for x in range(1,len(image)-1):
        for y in range(1,len(image[x])-1):
            # If we have a white pixel
            if image[x][y] == 1:
                neighbours = find_neighbours(x,y,image)
                valid = True
                if neighbours in validIntersection:
                    intersections.append((y,x))
    # Filter intersections to make sure we don't count them twice or ones that are very close together
    for point1 in intersections:
        for point2 in intersections:
            if (((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2) < 10**2) and (point1 != point2):
                intersections.remove(point2)
    # Remove duplicates
    intersections = list(set(intersections))
    return intersections

def skeleton_endpoints(skel, perim_limit=0):
    # Make our input nice, possibly necessary.
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)

    # Apply the convolution.
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)

    # Look through to find the value of 11.
    # This returns a mask of the endpoints, but if you
    # just want the coordinates, you could simply
    # return np.where(filtered==11)
    out = np.zeros_like(skel)
    out[np.where(filtered==11)] = 1
    
    # get points coordinates list
    ends_arr = np.nonzero(out)
    ends_list = []
    perim_list = []
    for y, x in zip(ends_arr[0], ends_arr[1]):
        if (x>=perim_limit) and (x<=skel.shape[1]-perim_limit) and (y>=perim_limit) and (y<=skel.shape[0]-perim_limit):
            ends_list.append((x,y))
        else:
            perim_list.append((x,y))
    return ends_list, perim_list

def get_path(cost_array, start_point, end_point, corr_i=0, corr_j=0, show=False):
    # get indexes
    i_start = start_point[1] - corr_i
    j_start = start_point[0] - corr_j
    i_stop = end_point[1] - corr_i
    j_stop = end_point[0] - corr_j
    
    # show
    if show:
        print((i_start, j_start), (i_stop, j_stop))
        plt.imshow(cost_array), plt.show()
    
    # calculate least cost path
    indices, weight = route_through_array(
        cost_array,
        (i_start, j_start),
        (i_stop, j_stop),
        geometric=True,
        fully_connected=True
        )
    indices = np.array(indices).T
    path = np.zeros_like(cost_array)
    path[indices[0], indices[1]] = 1
    return path, weight

def get_path_angles(p1, p2, skel_labels, path, corr_i=0, corr_j=0, pad_size=5):
    
    # get roads angles
    seg1_angle, seg2_angle = get_roads_angles(p1, p2, skel_labels, corr_i=corr_i, corr_j=corr_j, pad_size=pad_size)
    
    # get path angle
    path_window = np.pad(path, pad_size, constant_values=0)
    path_ends, _ = skeleton_endpoints(path_window, perim_limit=0)
    path_point1, path_point2 = path_ends[0], path_ends[-1]
    
    # get which point is on which segment
    if path_point1 == (p1[0]-corr_j+pad_size, p1[1]-corr_i+pad_size):
        path_seg1_point = path_point1
        path_seg2_point = path_point2
    else:
        path_seg1_point = path_point2
        path_seg2_point = path_point1
    
    # get angles of path -> p1 and path -> p2
    p1p2_angle = np.arctan2(-(path_seg1_point[1]-path_seg2_point[1]), path_seg1_point[0]-path_seg2_point[0])
    p2p1_angle = np.arctan2(-(path_seg2_point[1]-path_seg1_point[1]), path_seg2_point[0]-path_seg1_point[0])
    
    # get angles between path and segments
    seg1_path_angle = np.abs(np.rad2deg(p1p2_angle - seg1_angle))
    seg2_path_angle = np.abs(np.rad2deg(p2p1_angle - seg2_angle))

    print()
    print(path_ends)
    print(np.rad2deg(p1p2_angle), np.rad2deg(p2p1_angle), seg1_path_angle, seg2_path_angle)

    return seg1_path_angle, seg2_path_angle

def get_roads_angles(p1, p2=None, skel_labels=None, corr_i=0, corr_j=0, pad_size=5):
    # get position of segment 1 extremities
    seg1_val = skel_labels[p1[1] - corr_i, p1[0] - corr_j]
    seg1_window = np.pad(np.where(skel_labels==seg1_val, 1, 0), pad_size, constant_values=0)
    seg1_ends, _ = skeleton_endpoints(seg1_window, perim_limit=0)
    seg1_point1, seg1_point2 = seg1_ends[0], seg1_ends[-1]
    # make sure points are in right order
    if seg1_point2 != (p1[0]-corr_j+pad_size, p1[1]-corr_i+pad_size):
        seg1_point1, seg1_point2 = seg1_point2, seg1_point1
    
    # check if intersection close to point
    intersections = skeleton_intersections(np.uint8(seg1_window))+[seg1_point1]
    if len(intersections)>1:
        distances = [distance.euclidean(seg1_point2, p) for p in intersections]
        seg1_point1 = intersections[np.argmin(distances)]
    
    # get segments angle
    seg1_angle = np.arctan2(-(seg1_point2[1]-seg1_point1[1]), seg1_point2[0]-seg1_point1[0])
    
    # get position of segment 2 extremities
    if p2 is not None:
        seg2_val = skel_labels[p2[1] - corr_i, p2[0] - corr_j]
        seg2_window = np.pad(np.where(skel_labels==seg2_val, 1, 0), pad_size, constant_values=0)
        seg2_ends, _ = skeleton_endpoints(seg2_window, perim_limit=0)
        seg2_point1, seg2_point2 = seg2_ends[0], seg2_ends[-1]
        # make sure points are in right order
        if seg2_point2 != (p2[0]-corr_j+pad_size, p2[1]-corr_i+pad_size):
            seg2_point1, seg2_point2 = seg2_point2, seg2_point1
        
        # check if intersection close to point
        intersections = skeleton_intersections(np.uint8(seg2_window))+[seg2_point1]
        if len(intersections)>1:
            distances = [distance.euclidean(seg2_point2, p) for p in intersections]
            seg2_point1 = intersections[np.argmin(distances)]
        
        # get segments angle
        seg2_angle = np.arctan2(-(seg2_point2[1]-seg2_point1[1]), seg2_point2[0]-seg2_point1[0])
    else:
        seg2_angle = None

    return seg1_angle, seg2_angle

def get_cost_array(point_1, angle_1, point_2=None, angle_2=None, corr_i=0, corr_j=0, win_shape=None):
    
    pe = np.ones(win_shape)
    point_list = [point_1] if point_2 is None else [point_1, point_2]
    angle_list = [angle_1] if angle_2 is None else [angle_1, angle_2]
    
    for p, a in zip(point_list, angle_list):
        # directional ecoding
        pe_gauss = np.zeros(win_shape) + 0.01
        pe_gauss[p[1]-corr_i,p[0]-corr_j] = 1
        pe_gauss_shape_0 = pe_gauss.shape[0] if pe_gauss.shape[0]%2 == 1 else pe_gauss.shape[0]-1
        pe_gauss_shape_1 = pe_gauss.shape[1] if pe_gauss.shape[1]%2 == 1 else pe_gauss.shape[1]-1
        pe_gauss = cv2.GaussianBlur(pe_gauss, (pe_gauss_shape_0,pe_gauss_shape_1), 0)
        pe_gauss = ((pe_gauss - pe_gauss.min()) / (pe_gauss.max() - pe_gauss.min()))
        
        # get directional encoding
        pe_h = np.ones(pe_gauss.shape)*(np.arange(pe_gauss.shape[0])[:,None] - (p[1]-corr_i))
        pe_v = np.ones(pe_gauss.shape)*(np.arange(pe_gauss.shape[1])[:,None].T - (p[0]-corr_j))
        pe_dir = (np.cos(a) * pe_v - np.sin(a) * pe_h)
        pe_weight = (np.cos(a+np.pi/2) * pe_v - np.sin(a+np.pi/2) * pe_h)
        
        # remove path possibilities behind relative position (no backward paths)
        if point_2 is None:
            pe_dir = np.abs(pe_dir)
        pe_weight[pe_dir<0] = 0
        pe_gauss[pe_dir<0] = 0
        
        # Normalize directional cost
        pe_weight = np.abs(pe_weight)
        pe_weight = ((pe_weight - pe_weight.min()) / (pe_weight.max() - pe_weight.min()))*-1 + 1
        pe_weight[pe_dir<0] = 0
        pe_weight = np.power(pe_weight, 2)
        
        # get final segment path cost
        pe *= pe_gauss + pe_weight
    if pe.max() > 0:
        pe /= pe.max()
    pe = 1 - pe
    
    return pe


def execute(exp_name: str, save_folder: str, pred_dir: str, save_dir: str, annotations_index_path: str
            , overlap=0.20, img_size=3000, img_perimeter_limit=30, prob_threshold=0.5, max_dist_seg2seg_meters=125
            , max_dist_seg2graph_meters=50, min_segment_area_preprocess_m2=50, min_segment_area_postprocess_m2=1000
            , width_fit_size=30, nodata_val=255, new_segments_value=2):

    if not os.path.exists(annotations_index_path):
        raise ValueError('annotations_index_path does not exist.')
    if not os.path.exists(pred_dir):
        raise ValueError('pred_dir does not exist.')
    if not os.path.exists(save_dir + '/' + save_folder):
        os.makedirs(save_dir + '/' + save_folder)

    # Get predictions and save paths
    pred_paths: list = glob(pred_dir + exp_name + '/*.tif')
    save_paths = [save_dir + save_folder + '/' + pred_path.split('\\')[-1] for pred_path in pred_paths]
    save_paths = ['_'.join(save_path.split('_')[:-3]) + '_mask.tif' for save_path in save_paths] # replace model name by "_mask"

    # Convolution kernels
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    uniform_kernel = np.ones((3,3)).astype(np.uint8)

    # Postprocess predictions
    for pred_path, save_path in tqdm(zip(pred_paths, save_paths)):
        if os.path.exists(save_path):  # Skip if already done
            continue

        # Console log
        f_name = pred_path.split('\\')[-1]
        print(f"Reading {f_name}...")

        # Open raster files
        pred_file = rasterio.open(pred_path)
        kwds = pred_file.profile
        kwds['count'] = 1
        kwds['dtype'] = np.uint8
        save_file = rasterio.open(save_path, 'w+', **kwds)

        # Get parameters in pixels number
        res = np.abs(pred_file.transform[0])
        max_dist_seg2seg = max_dist_seg2seg_meters//res
        max_dist_seg2graph = max_dist_seg2graph_meters//res
        min_segment_area_preprocess = min_segment_area_preprocess_m2//res//res
        min_segment_area_postprocess = min_segment_area_postprocess_m2//res//res

        # Loop in images to create dataset
        idx_i = 0
        last_i, done_i = False, False
        while not done_i:
            idx_j = 0
            last_j, done_j = False, False

            # Console log
            print("Completion : " + str(idx_i / pred_file.shape[0] * 100))

            while not done_j:
                ''' 1. Read arrays and get skeleton '''
                # prepare sub arrays
                win = windows.Window(idx_j, idx_i, img_size, img_size)
                sub_prob = pred_file.read(window=win)
                sub_prob = np.true_divide(sub_prob[0], sub_prob[-1], out=np.zeros_like(sub_prob[0]), where=sub_prob[-1]!=0)
                sub_pred = np.where(sub_prob>=prob_threshold, 1, 0).astype(np.uint8)

                sub_new_pred = np.copy(sub_pred).astype(np.uint8)
                if (sub_pred != 0).any():

                    ''' 2. Remove small artefacts '''
                    # get connected components on map
                    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(sub_pred)

                    # remove small objects
                    for i in range(1,numLabels):
                        if stats[i, cv2.CC_STAT_AREA] < min_segment_area_preprocess:
                            sub_pred[labels==i] = 0

                    # skeletonize prediction and ground truth
                    sub_pred_ske = get_skeleton(sub_pred)


                    ''' 3. Connect close segments '''
                    # get end points in prediction
                    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(sub_pred_ske)
                    ends, _ = skeleton_endpoints(sub_pred_ske, perim_limit=img_perimeter_limit)

                    # Compute least cost path matrix
                    weights_matrix = np.ones((len(ends),len(ends)))*65535
                    paths_list = [[] for j in range(weights_matrix.shape[1])]
                    paths_list = [paths_list for i in range(weights_matrix.shape[0])]
                    indexes_list = copy.deepcopy(paths_list)
                    for i in tqdm(range(weights_matrix.shape[0])):
                        for j in range(weights_matrix.shape[1]):

                            # get points candidates
                            point_1, point_2 = ends[i], ends[j]
                            dist_seg2seg = distance.euclidean(point_1, point_2)

                            # compute diagonal matrix only
                            if i >= j:
                                # weights_matrix[i,j] = weights_matrix[j,i]
                                continue
                            # skip if on same segment or inter points distance is too large
                            elif dist_seg2seg < max_dist_seg2seg and labels[point_1[1], point_1[0]] != labels[point_2[1], point_2[0]]:

                                # get search window indexes
                                idx_i_min = int(max(0, min(point_1[1]-max_dist_seg2graph, point_2[1]-max_dist_seg2graph)))
                                idx_j_min = int(max(0, min(point_1[0]-max_dist_seg2graph, point_2[0]-max_dist_seg2graph)))
                                idx_i_max = int(min(sub_prob.shape[0], max(point_1[1]+max_dist_seg2graph, point_2[1]+max_dist_seg2graph)))
                                idx_j_max = int(min(sub_prob.shape[1], max(point_1[0]+max_dist_seg2graph, point_2[0]+max_dist_seg2graph)))

                                # get angles of segments to connect
                                sub_labels = labels[idx_i_min:idx_i_max, idx_j_min:idx_j_max]
                                angle_1, angle_2 = get_roads_angles(point_1, point_2, sub_labels, corr_i=idx_i_min, corr_j=idx_j_min)

                                # get cost array
                                cost_arr = get_cost_array(point_1, angle_1, point_2, angle_2, idx_i_min, idx_j_min, sub_labels.shape)
                                cost_at_p1, cost_at_p2 = cost_arr[point_1[1]-idx_i_min,point_1[0]-idx_j_min], cost_arr[point_2[1]-idx_i_min,point_2[0]-idx_j_min]
                                if cost_at_p1<1 and cost_at_p2<1:
                                    cost_prob_arr = cv2.blur(1-sub_prob[idx_i_min:idx_i_max, idx_j_min:idx_j_max], (5,5))
                                    cost_arr += cost_prob_arr/2
                                    cost_arr[sub_pred_ske[idx_i_min:idx_i_max, idx_j_min:idx_j_max]>=1] = 0

                                    # compute least cost path
                                    path, weight = get_path(cost_arr, point_1, point_2, corr_i=idx_i_min, corr_j=idx_j_min)

                                    # add to weights matrix and paths list
                                    if (path.sum() < max_dist_seg2seg) and (path.sum() > weight):
                                        weights_matrix[i,j] = weight
                                        paths_list[i][j] = path
                                        indexes_list[i][j] = [idx_i_min, idx_j_min, idx_i_max, idx_j_max]

                    # Get minimum weight paths
                    for i in range(weights_matrix.shape[0]):
                        neighbor_index = np.argmin(weights_matrix[i,:]) if weights_matrix[i,:].min() != 65535 else None

                        # get path and clip indexes
                        if neighbor_index is not None:
                            path = paths_list[i][neighbor_index]
                            idx_i_min, idx_j_min, idx_i_max, idx_j_max = indexes_list[i][neighbor_index]

                            # add correction to skeleton
                            sub_pred_ske[idx_i_min:idx_i_max, idx_j_min:idx_j_max] = 2*path + sub_pred_ske[idx_i_min:idx_i_max, idx_j_min:idx_j_max]
                            sub_pred_ske[sub_pred_ske>2] = 2

                            # Visualization
                            # plt.subplot(121), plt.imshow(path)
                            # plt.subplot(122), plt.imshow(labels[idx_i_min:idx_i_max, idx_j_min:idx_j_max])
                            # plt.show()


                    ''' 4. Connect ends to network '''
                    # Check if there are other possible connections
                    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(sub_pred_ske)
                    ends, _ = skeleton_endpoints(sub_pred_ske, perim_limit=img_perimeter_limit)
                    for end_ in ends:
                        # get skel pred  and cost array
                        idx_i_min = int(max(0, end_[1]-max_dist_seg2graph))
                        idx_j_min = int(max(0, end_[0]-max_dist_seg2graph))
                        idx_i_max = int(min(sub_pred.shape[0], end_[1]+max_dist_seg2graph))
                        idx_j_max = int(min(sub_pred.shape[1], end_[0]+max_dist_seg2graph))

                        # get connected component of skel to evaluate
                        eval_labels = np.copy(labels[idx_i_min:idx_i_max, idx_j_min:idx_j_max])
                        eval_labels[eval_labels==eval_labels[end_[1]-idx_i_min,end_[0]-idx_j_min]] = 0
                        if (eval_labels>0).sum() > 2:

                            # get angles of segments to connect
                            sub_labels = labels[idx_i_min:idx_i_max, idx_j_min:idx_j_max]
                            angle_1, _ = get_roads_angles(end_, None, sub_labels, corr_i=idx_i_min, corr_j=idx_j_min)

                            # get cost array
                            cost_arr = get_cost_array(end_, angle_1, corr_i=idx_i_min, corr_j=idx_j_min, win_shape=sub_labels.shape)
                            cost_arr *= 5
                            cost_arr += (1 - sub_prob[idx_i_min:idx_i_max, idx_j_min:idx_j_max])
                            cost_arr = (cost_arr - cost_arr.min()) / (cost_arr.max() - cost_arr.min()) + 0.001
                            cost_arr[sub_pred_ske[idx_i_min:idx_i_max, idx_j_min:idx_j_max]>=1] = 1e-6
                            # plt.imshow(cost_arr), plt.show()

                            # get least cost path
                            path, weight = None, 1e30
                            rows, cols = np.where(eval_labels)
                            row, col = None, None
                            for row_, col_ in zip(rows, cols):
                                path_, weight_ = get_path(cost_arr, end_, (col_+idx_j_min,row_+idx_i_min), corr_i=idx_i_min, corr_j=idx_j_min)
                                if weight_ < 1e-1:
                                    path = None
                                    break # already connected
                                elif weight_ < weight :
                                    weight, path, row, col = weight_, path_, row_, col_
                            if path is None:
                                continue

                            # make sure path is a skeleton or errors can happen when finding end points
                            path = skeletonize(path)
                            path[end_[0]-idx_j_min, end_[1]-idx_i_min] = 1
                            if row or col:
                                path[row, col] = 1
                            else:
                                raise ValueError("No path found")

                            # add correction to skeleton
                            angle_1, angle_2 = get_path_angles((end_[0]-idx_j_min, end_[1]-idx_i_min), (col, row), labels[idx_i_min:idx_i_max, idx_j_min:idx_j_max], path)
                            diff_to_90_1, diff_to_90_2 = np.abs((angle_1 + 180)%180 - 90), np.abs((angle_2 + 180)%180 - 90)
                            if (diff_to_90_1 > 45) or (diff_to_90_2 > 45):
                                sub_pred_ske[idx_i_min:idx_i_max, idx_j_min:idx_j_max] = 2*path+sub_pred_ske[idx_i_min:idx_i_max, idx_j_min:idx_j_max]
                                sub_pred_ske[sub_pred_ske>2] = 2

                                # print(path.sum(), weight, diff_to_90_1, diff_to_90_2)
                                # plt.subplot(131), plt.imshow(cost_arr)
                                # plt.subplot(132), plt.imshow(path)
                                # plt.subplot(133), plt.imshow(labels[idx_i_min:idx_i_max, idx_j_min:idx_j_max])
                                # plt.show()

                    ''' 5. Find width of new segments '''
                    numLabels, new_paths, stats, centroids = cv2.connectedComponentsWithStats((sub_pred_ske>1).astype(np.uint8))
                    for i in range(1,numLabels):
                        # get window of interest
                        i_start = max(0, int(stats[i, cv2.CC_STAT_TOP])-width_fit_size)
                        j_start = max(0, int(stats[i, cv2.CC_STAT_LEFT])-width_fit_size)
                        i_stop = min(sub_pred_ske.shape[0], int(stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT])+width_fit_size)
                        j_stop = min(sub_pred_ske.shape[0], int(stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH])+width_fit_size)

                        # get prediction and correction in zone
                        pred_zone = sub_pred[i_start:i_stop, j_start:j_stop]
                        corr_zone = sub_pred_ske[i_start:i_stop, j_start:j_stop]
                        fit_arr = ((pred_zone*corr_zone)>0).astype(np.uint8)

                        # get number of dilation iterations
                        iter_count = 0
                        score = 0
                        one_more_try = True
                        search = True
                        target = pred_zone.flatten()
                        while search:
                            fit_arr = cv2.dilate(fit_arr, dilate_kernel, iterations=1)
                            input_test = fit_arr.flatten()
                            score_test = accuracy_score(target, input_test)
                            iter_count += 1
                            if score_test > score:
                                score = score_test
                                one_more_try = True
                            else:
                                if one_more_try:
                                    one_more_try = False
                                else:
                                    search = False
                        iter_count = iter_count-2

                        # dilate new segment
                        corr_zone = (((1-pred_zone)*corr_zone)>0).astype(np.uint8)
                        corr_zone = cv2.dilate(corr_zone, dilate_kernel, iterations=iter_count)

                        # add correction to prediction
                        sub_new_pred[i_start:i_stop, j_start:j_stop] = corr_zone+pred_zone
                        sub_new_pred[sub_new_pred>1] = 1

                    # identify correction with new number
                    added_segment = (sub_new_pred-sub_pred)>0
                    sub_new_pred[added_segment] = 2

                    ''' 7. Smooth new prediction '''
                    ## majority filter
                    #sub_new_pred = cv2.morphologyEx(sub_new_pred, cv2.MORPH_CLOSE, uniform_kernel)
                    ## dilation
                    #sub_new_pred = cv2.dilate(sub_new_pred, dilate_kernel, iterations=1)
                    ## erosion
                    #sub_new_pred = cv2.erode(sub_new_pred, erode_kernel, iterations=1)

                    ''' 8. Remove small artefacts '''
                    # get connected components on map
                    sub_new_pred = np.uint8(sub_new_pred)
                    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(sub_new_pred)
                    # remove small objects
                    for i in range(1,numLabels):
                        if stats[i, cv2.CC_STAT_AREA] < min_segment_area_postprocess:
                            sub_new_pred[labels==i] = 0

                    ''' 9. Save to new prediction '''
                    pred_to_merge = save_file.read(window=win)[0]
                    sub_new_pred[pred_to_merge==1] = 1
                    sub_new_pred[pred_to_merge==2] = new_segments_value
                    save_file.write(sub_new_pred, indexes=1, window=win)
                else:
                    # Save to raster
                    save_file.write(sub_new_pred, indexes=1, window=win)
                # calculate new idx_j
                if last_j:
                    done_j = True
                else:
                    idx_j += int(img_size * (1.0-overlap))
                    if idx_j+img_size >= pred_file.shape[1]:
                        idx_j = pred_file.shape[1] - img_size
                        last_j = True
            # calculate new idx_i
            if last_i:
                done_i = True
            else:
                idx_i += int(img_size * (1.0-overlap))
                if idx_i+img_size >= pred_file.shape[0]:
                    idx_i = pred_file.shape[0] - img_size
                    last_i = True

        # Clip to minimum annotated zone for comparison
        pred = save_file.read()[0]
        pred[pred==new_segments_value] = 1
        pred, kwds = clip_to_min_extent(annotations_index_path, pred, pred_path.split('\\')[-1], save_file.crs, save_file.profile, nodata_val)

        # Save to raster
        save_file.close()
        with rasterio.open(save_path, 'w', **kwds) as save_file:
            save_file.write_band(1, pred)