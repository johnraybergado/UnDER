import os
from io import StringIO
import time
# import math
# from collections import OrderedDict
# import glob
import subprocess
import re
from typing import cast
import json
import pickle
from PIL import Image
import numpy as np
# import cv2 as cv
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
# import skimage.io
from skimage import transform
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Polygon, Point, shape, MultiPolygon
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from osgeo import gdal, gdalconst
# from osgeo import gdalconst
# from bs4 import BeautifulSoup
# from pyoints import (transformation,
#                      projection,
#                      georecords,
#                      storage)
# from pyoints.grid import grid
import rasterio
import rasterio.features
# from rasterio.merge import merge
# from rasterio.enums import Resampling
# from rasterio.fill import fillnodata
# from rasterio.features import dataset_features
import psycopg2
import psycopg2.extras
from psycopg2 import Error
from datasets.tmp_dataset import TMPDataset
from models.PASMnet import *
# from disparity.rectify import rectify_stereopair
# from disparity.camera_params_UG import init_img_params, estimate_GSD
from under_pipeline.rectify import rectify_stereopair
from under_pipeline.camera_params_UG import init_img_params, estimate_GSD


def write_array(tf_imgarray, fname):
    rows, cols, nbands = tf_imgarray.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(fname, cols, rows, nbands, gdalconst.GDT_Byte)
    imgarray_byte = (tf_imgarray*255).astype(np.uint8)
    for i in range(nbands):
        outdata.GetRasterBand(i+1).WriteArray(imgarray_byte[:, :, i])
    outdata.FlushCache()
    outdata = None

def write_array_float(imgarray, fname):
    rows, cols, nbands = imgarray.shape
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(fname, cols, rows, nbands, gdalconst.GDT_Float32)
    # imgarray_byte = (tf_imgarray*255).astype(np.uint8)
    for i in range(nbands):
        outdata.GetRasterBand(i+1).WriteArray(imgarray[:, :, i])
    outdata.FlushCache()
    outdata = None

def write_disparity_PASMNet(height_snap, width_snap, filenames,
                            data_path, model_path):
    DEVICE = 'cuda:0'
    # MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\PASMnet_KITTI2015_epoch80.pth'
    # MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\epoch79_loss_3_7_epe_5_8_d3_18_2.pth.tar'
    # MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\epoch20_loss_0_19_epe_9_8_d3_22_3.pth.tar'
    DATA_DIR = 'tmp'
    # DATA_PATH = r'D:\\Workspace\\UT_postdoc\80_tools\\00_dronemappy\\dronemappy\\tmp'
    # MAX_DISPARITY = 192
    # MAX_DISPARITY = 512
    MAX_DISPARITY = 0

    test_set = TMPDataset(datapath=data_path,
                          list_filename=filenames,
                          training=False,
                          height_snap=height_snap,
                          width_snap=width_snap)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    net = PASMnet().to(DEVICE)
    if model_path.endswith("tar"):
        ckpt = torch.load(model_path)["state_dict"]
    else:
        ckpt = torch.load(MODEL_PATH)
    net.load_state_dict(ckpt)
    net.eval()
    cudnn.benchmark = True
    if not os.path.exists(data_path + os.path.sep + 'left_disp' + os.path.sep + DATA_DIR + '_' + str(MAX_DISPARITY)):
        os.mkdir(data_path + os.path.sep + 'left_disp' + os.path.sep + DATA_DIR + '_' + str(MAX_DISPARITY))

    with torch.no_grad():
        for iteration, data in enumerate(test_loader):
            img_left, img_right = data['left'].to(DEVICE), data['right'].to(DEVICE)

            top_pad = int(data['top_pad'].data.cpu())
            right_pad = int(data['right_pad'].data.cpu())

            # disp = net(img_left, img_right, max_disp=MAX_DISPARITY)
            # disp = torch.clamp(disp[:, :, top_pad:, :-right_pad].squeeze().data.cpu(), 0).numpy()
            # disp_fname = DATA_PATH + os.path.sep + 'left_disp' + os.path.sep + DATA_DIR + '_' + str(MAX_DISPARITY) + '/' +\
            #              test_loader.dataset.left_filenames[iteration][-13:]
            # disp = np.expand_dims(disp, axis=-1)
            # write_array_float(disp, disp_fname)
            disp, mask = net(img_left, img_right, max_disp=MAX_DISPARITY)
            disp = torch.clamp(disp[:, :, top_pad:, :-right_pad].squeeze().data.cpu(), 0).numpy()
            mask = torch.clamp(mask[:, :, top_pad:, :-right_pad].squeeze().data.cpu(), 0).numpy()
            disp_fname = data_path + os.path.sep + 'left_disp' + os.path.sep + DATA_DIR + '_' + str(MAX_DISPARITY) + '/' +\
                         test_loader.dataset.left_filenames[iteration][-13:]
            mask_fname = data_path + os.path.sep + 'left_disp' + os.path.sep + DATA_DIR + '_' + str(MAX_DISPARITY) + '/' +\
                         test_loader.dataset.left_filenames[iteration][-13:-4] + "_mask.tif"
            disp = np.expand_dims(disp, axis=-1)
            mask = np.expand_dims(mask, axis=-1)
            write_array_float(disp, disp_fname)
            write_array_float(mask, mask_fname)


def write_disparity_SGM(imgpath1, imgpath2):
    img1 = Image.open(imgpath1).convert("L")
    out_img1 = imgpath1.replace(TMP_FNAME, "grayscale")
    img1.save(out_img1)
    img2 = Image.open(imgpath2).convert("L")
    out_img2 = imgpath2.replace(TMP_FNAME, "grayscale")
    img2.save(out_img2)
    command = "pandora {} {}".format(SGM_CONFIG, SGM_OUT)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)


def warp_left_disparity(disp_array, output_shape, rect_params):
    """
    Warp disparity image back to original image space.

    :param disp_array: disparity image numpy array
    :param output_shape: shape of the original image space
    :param rect_params: dictionary containing rectification parameters
    :return: warped image array and dictionary of left and right mappings
    """
    K = rect_params["K"]
    H1 = rect_params["H1"]
    H2 = rect_params["H2"]
    H_shift = rect_params["H_shift"]
    c = K[0][0]
    x_c_orig = K[0, 2]
    y_c_orig = K[1, 2]
    left_mapping = H1 @ H_shift
    right_mapping = H2 @ H_shift

    IC = np.array([x_c_orig, y_c_orig, 1])
    IC = np.linalg.inv(left_mapping) @ IC
    x_c, y_c = IC[0] / IC[-1], IC[1] / IC[-1]

    RC = np.array([x_c_orig, y_c_orig, 1])
    RC = np.linalg.inv(right_mapping) @ RC
    x_c_r, y_c_r = RC[0] / RC[-1], RC[1] / RC[-1]
    px_c = x_c - x_c_r

    tform = transform.ProjectiveTransform(matrix=np.linalg.inv(left_mapping))
    disp_array_orig = transform.warp(disp_array, tform, output_shape=output_shape)
    mappings = {"left": left_mapping, "right": right_mapping}
    return disp_array_orig, mappings


def calc_P_matrix(img_params):
    K = img_params["camera_matrix"]
    R = img_params["rotation_matrix"]
    C = np.array([[img_params["X"]],
                  [img_params["Y"]],
                  [img_params["Z"]]], dtype=np.float64)
    t = -R@C
    R_t = np.zeros((3, 4), dtype=np.float64)
    R_t[:, :-1] = R
    R_t[:, -1] = t[:, 0]
    P = K@R_t
    return P


# # method from https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py
# # Initialize consts to be used in iterative_LS_triangulation()
# iterative_LS_triangulation_C = -np.eye(2, 3)


# def iterative_LS_triangulation(u1, P1, u2, P2, tolerance=3.e-5):
#     """
#     Iterative (Linear) Least Squares based triangulation.
#     From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997.
#     Relative speed: 0.025

#     (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
#     (u2, P2) is the second pair.
#     "tolerance" is the depth convergence tolerance.

#     Additionally returns a status-vector to indicate outliers:
#         1: inlier, and in front of both cameras
#         0: outlier, but in front of both cameras
#         -1: only in front of second camera
#         -2: only in front of first camera
#         -3: not in front of any camera
#     Outliers are selected based on non-convergence of depth, and on negativity of depths (=> behind camera(s)).

#     u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
#     """
#     A = np.zeros((4, 3))
#     b = np.zeros((4, 1))

#     # Create array of triangulated points
#     x = np.empty((4, len(u1)));
#     x[3, :].fill(1)  # create empty array of homogenous 3D coordinates
#     x_status = np.empty(len(u1), dtype=int)

#     # Initialize C matrices
#     C1 = np.array(iterative_LS_triangulation_C)
#     C2 = np.array(iterative_LS_triangulation_C)

#     for xi in range(len(u1)):
#         # Build C matrices, to construct A and b in a concise way
#         C1[:, 2] = u1[xi, :]
#         C2[:, 2] = u2[xi, :]

#         # Build A matrix
#         A[0:2, :] = C1.dot(P1[0:3, 0:3])  # C1 * R1
#         A[2:4, :] = C2.dot(P2[0:3, 0:3])  # C2 * R2

#         # Build b vector
#         b[0:2, :] = C1.dot(P1[0:3, 3:4])  # C1 * t1
#         b[2:4, :] = C2.dot(P2[0:3, 3:4])  # C2 * t2
#         b *= -1

#         # Init depths
#         d1 = d2 = 1.

#         for i in range(10):  # Hartley suggests 10 iterations at most
#             # Solve for x vector
#             # x_old = np.array(x[0:3, xi])    # TODO: remove
#             cv.solve(A, b, x[0:3, xi:xi + 1], cv.DECOMP_SVD)

#             # Calculate new depths
#             d1_new = P1[2, :].dot(x[:, xi])
#             d2_new = P2[2, :].dot(x[:, xi])

#             # Convergence criterium
#             # print i, d1_new - d1, d2_new - d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
#             # print i, (d1_new - d1) / d1, (d2_new - d2) / d2, (d1_new > 0 and d2_new > 0)    # TODO: remove
#             # print i, np.sqrt(np.sum((x[0:3, xi] - x_old)**2)), (d1_new > 0 and d2_new > 0)    # TODO: remove
#             ##print i, u1[xi, :] - P1[0:2, :].dot(x[:, xi]) / d1_new, u2[xi, :] - P2[0:2, :].dot(x[:, xi]) / d2_new    # TODO: remove
#             # print bool(i) and ((d1_new - d1) / (d1 - d_old), (d2_new - d2) / (d2 - d1_old), (d1_new > 0 and d2_new > 0))    # TODO: remove
#             ##if abs(d1_new - d1) <= tolerance and abs(d2_new - d2) <= tolerance: print "Orig cond met"    # TODO: remove
#             if abs(d1_new - d1) <= tolerance and \
#                     abs(d2_new - d2) <= tolerance:
#                 # if i and np.sum((x[0:3, xi] - x_old)**2) <= 0.0001**2:
#                 # if abs((d1_new - d1) / d1) <= 3.e-6 and \
#                 # abs((d2_new - d2) / d2) <= 3.e-6: #and \
#                 # abs(d1_new - d1) <= tolerance and \
#                 # abs(d2_new - d2) <= tolerance:
#                 # if i and 1 - abs((d1_new - d1) / (d1 - d_old)) <= 1.e-2 and \    # TODO: remove
#                 # 1 - abs((d2_new - d2) / (d2 - d1_old)) <= 1.e-2 and \    # TODO: remove
#                 # abs(d1_new - d1) <= tolerance and \    # TODO: remove
#                 # abs(d2_new - d2) <= tolerance:    # TODO: remove
#                 break

#             # Re-weight A matrix and b vector with the new depths
#             A[0:2, :] *= 1 / d1_new
#             A[2:4, :] *= 1 / d2_new
#             b[0:2, :] *= 1 / d1_new
#             b[2:4, :] *= 1 / d2_new

#             # Update depths
#             # d_old = d1    # TODO: remove
#             # d1_old = d2    # TODO: remove
#             d1 = d1_new
#             d2 = d2_new

#         # Set status
#         x_status[xi] = (i < 10 and  # points should have converged by now
#                         (d1_new > 0 and d2_new > 0))  # points should be in front of both cameras
#         if d1_new <= 0: x_status[xi] -= 1
#         if d2_new <= 0: x_status[xi] -= 2

#     return x[0:3, :].T.astype(np.float64), x_status


def triangulate_left(img1, img2, orig_disp_array1, orig_mask_array1,
                     mappings, rect_params, img_params):
    """
    Triangulate using left disparity image.

    :param img1: file name of the left image
    :param img2: file name of the right image
    :param orig_disp_array1: left disparity image array
    :param mappings: dictonary containing left and right mappings
    :param rect_params: dictionary containing rectification parameters
    :param img_params: ordered dict containing image parameters
    :return:
    """
    K = rect_params["K"]
    imgparams1 = img_params[img1]
    imgparams2 = img_params[img2]
    R1 = imgparams1["rotation_matrix"]
    R2 = imgparams2["rotation_matrix"]
    Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
    Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
    c = K[0][0]
    # calculate image coordinates of overlap
    left_extent = imgparams1["geobounds"]
    right_extent = imgparams2["geobounds"]
    mean_z = imgparams1["mean_gcp_height"]
    left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
    right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
    # overlap = left_extent.intersection(right_extent)
    overlap: BaseGeometry = left_extent.intersection(right_extent)
    # If this ever isn't a Polygon, fail loudly so the assumption is explicit
    if not isinstance(overlap, Polygon):
        raise ValueError(f"Expected Polygon overlap, got {overlap.geom_type!r}")
    overlap = cast(Polygon, overlap)
    xx, yy = overlap.exterior.coords.xy
    xx = xx.tolist()
    yy = yy.tolist()
    overlap_coords_world = [(x, y, mean_z, 1) for x, y in zip(xx, yy)]
    overlap_coords_world = overlap_coords_world[:-1]
    # print(overlap_coords_world)
    t = -R1@Z1
    R_t = np.zeros((3, 4), dtype=np.float64)
    R_t[:, :-1] = R1
    R_t[:, -1] = t
    P = K@R_t
    overlap_coords_image = [P@np.array(i) for i in overlap_coords_world]
    overlap_coords_image = [i / i[-1] for i in overlap_coords_image]
    overlap_bounds = Polygon([i[:-1] for i in overlap_coords_image])
    overlap_bounds = overlap_bounds.buffer(-OVERLAP_BOUNDS_BUFFER)
    mask = rasterio.features.rasterize([overlap_bounds], out_shape=orig_disp_array1.shape)
    # add mask from validity/occlusion
    mask = mask * orig_mask_array1
    x_c_orig = K[0, 2]
    y_c_orig = K[1, 2]
    left_mapping = mappings["left"]
    right_mapping = mappings["right"]
    ny, nx = orig_disp_array1.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    xv = xv[mask==1]
    yv = yv[mask==1]
    disp_values = orig_disp_array1[mask==1]
    third_column = np.ones(xv.shape)
    left_data = np.stack((xv, yv, third_column, disp_values))
    right_data = left_data.copy()
    # map to warped space of left image
    right_data[0:3, :] = np.linalg.inv(left_mapping) @ right_data[0:3, :]
    right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
    # add x-disparity
    right_data[0:1, :] = right_data[0:1, :] - right_data[-1:, :]
    # map to orig space of right image
    right_data[0:3, :] = right_mapping @ right_data[0:3, :]
    right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
    left_coords = left_data[0:2]
    right_coords = right_data[0:2]
    del left_data, right_data
    left_coords = np.concatenate((left_coords, np.full((1, left_coords.shape[1]), c)), axis=0)
    right_coords = np.concatenate((right_coords, np.full((1, right_coords.shape[1]), c)), axis=0)
    # shift coordinate based on principal point coordinate
    left_coords[0] = left_coords[0] - x_c_orig
    left_coords[1] = left_coords[1] - y_c_orig
    right_coords[0] = right_coords[0] - x_c_orig
    right_coords[1] = right_coords[1] - y_c_orig
    n_points = left_coords.shape[1]
    idx = 0
    # print(left_coords)
    # print(right_coords)
    t2 = -R2@Z2
    R_t2 = np.zeros((3, 4), dtype=np.float64)
    R_t2[:, :-1] = R2
    R_t2[:, -1] = t2
    P2 = K@R_t2
    P1 = P

    # # use iterative LS
    # # normalize image coordinates
    # left_coords[0] = left_coords[0] / nx
    # left_coords[1] = left_coords[1] / ny
    # right_coords[0] = left_coords[0] / nx
    # right_coords[1] = left_coords[1] / ny
    # points_3D = cv.triangulatePoints(P1, P2, left_coords[:2, ], right_coords[:2, ])
    # points_3D = points_3D / points_3D[-1:, ]
    # points_3D = points_3D[:-1, :]
    # points_3D = points_3D.T
    # print(points_3D)
    # points_3D, _ = iterative_LS_triangulation(left_coords[:2, ].T, P1, right_coords[:2, ].T, P2)
    # print(_)
    # print(points_3D)
    # points_3D_RGB = np.zeros((n_points, 6), dtype=np.float64)
    # points_3D_RGB[:, 0:3] = points_3D
    
    # perform triangulation
    points_3D_RGB = np.zeros((n_points, 6), dtype=np.float64)
    for i in range(n_points):
        r = R1.T @ left_coords[:, i]
        s = R2.T @ right_coords[:, i]
        b = np.array([[np.dot((Z2 - Z1), r)],
                      [np.dot((Z2 - Z1), s)]])
        A = np.array([[r.T @ r, -s.T @ r],
                      [r.T @ s, -s.T @ s]])
        params = np.linalg.solve(A, b)
        F = Z1 + params[0, 0] * r
        G = Z2 + params[1, 0] * s
        point_coords = (F + G) / 2
        points_3D_RGB[i, 0:3] = point_coords
        if idx % 100000 == 0:
            t = time.time()
            t_mins = (t - T0) / 60.0
            print("{} of {} points triangulated after {:.2f} mins".format(idx, n_points, t_mins))
        idx += 1
    return points_3D_RGB, mask


def triangulate_multi(base_image, side_images, disp_arrays, mask_arrays,
                      mappings_list, rect_params_list, img_params):
    """
    Triangulate using left disparity image.

    :param base_image: file name of the base image
    :param side_images: list containing file names of the side images
    :param disp_arrays: list of left disparity image arrays
    :param mappings_list: list of dictonaries containing left and right mappings
    :param rect_params_list: list of dictionaries containing rectification parameters
    :param img_params: ordered dict containing image parameters
    :return:
    """
    # calculate aggregated mask
    mask_list = []
    for idx, side_image in enumerate(side_images):
        K = rect_params_list[idx]["K"]
        imgparams1 = img_params[base_image]
        imgparams2 = img_params[side_image]
        R1 = imgparams1["rotation_matrix"]
        R2 = imgparams2["rotation_matrix"]
        Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
        Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
        c = K[0][0]
        # calculate image coordinates of overlap and add mask based on it
        left_extent = imgparams1["geobounds"]
        right_extent = imgparams2["geobounds"]
        mean_z = imgparams1["mean_gcp_height"]
        left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
        right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
        # overlap = left_extent.intersection(right_extent)
        overlap: BaseGeometry = left_extent.intersection(right_extent)
        # If this ever isn't a Polygon, fail loudly so the assumption is explicit
        if not isinstance(overlap, Polygon):
            raise ValueError(f"Expected Polygon overlap, got {overlap.geom_type!r}")
        overlap = cast(Polygon, overlap)
        xx, yy = overlap.exterior.coords.xy
        xx = xx.tolist()
        yy = yy.tolist()
        overlap_coords_world = [(x, y, mean_z, 1) for x, y in zip(xx, yy)]
        overlap_coords_world = overlap_coords_world[:-1]
        # print(overlap_coords_world)
        t = -R1@Z1
        R_t = np.zeros((3, 4), dtype=np.float64)
        R_t[:, :-1] = R1
        R_t[:, -1] = t
        P = K@R_t
        overlap_coords_image = [P@np.array(i) for i in overlap_coords_world]
        overlap_coords_image = [i / i[-1] for i in overlap_coords_image]
        overlap_bounds = Polygon([i[:-1] for i in overlap_coords_image])
        overlap_bounds = overlap_bounds.buffer(-OVERLAP_BOUNDS_BUFFER)
        mask = rasterio.features.rasterize([overlap_bounds], out_shape=disp_arrays[idx].shape)
        mask = mask * mask_arrays[idx]
        mask_list.append(mask)

    aggregated_mask = np.ones(mask_list[0].shape)
    for i in range(len(mask_list)):
        aggregated_mask = aggregated_mask * mask_list[i]
    print(int(aggregated_mask.sum()))
    write_array_float(np.expand_dims(aggregated_mask, axis=-1), "aggregated_mask.tif")
    points_3D_RGB_multi = np.zeros((int((aggregated_mask==1).sum()), 6,
                                    len(side_images)), dtype=np.float64)
    for idx, side_image in enumerate(side_images):
        K = rect_params_list[idx]["K"]
        imgparams1 = img_params[base_image]
        imgparams2 = img_params[side_image]
        R1 = imgparams1["rotation_matrix"]
        R2 = imgparams2["rotation_matrix"]
        Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
        Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
        c = K[0][0]
        x_c_orig = K[0, 2]
        y_c_orig = K[1, 2]
        left_mapping = mappings_list[idx]["left"]
        right_mapping = mappings_list[idx]["right"]
        ny, nx = disp_arrays[idx].shape
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xv, yv = np.meshgrid(x, y)
        # print(xv.shape)
        # print(yv.shape)
        xv = xv[aggregated_mask==1]
        yv = yv[aggregated_mask==1]
        disp_values = disp_arrays[idx][aggregated_mask==1]
        # print(xv.shape)
        # print(yv.shape)
        third_column = np.ones(xv.shape)
        left_data = np.stack((xv, yv, third_column, disp_values))
        right_data = left_data.copy()
        # map to warped space of left image
        right_data[0:3, :] = np.linalg.inv(left_mapping) @ right_data[0:3, :]
        right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
        # add x-disparity
        right_data[0:1, :] = right_data[0:1, :] - right_data[-1:, :]
        # map to orig space of right image
        right_data[0:3, :] = right_mapping @ right_data[0:3, :]
        right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
        left_coords = left_data[0:2]
        right_coords = right_data[0:2]
        del left_data, right_data
        left_coords = np.concatenate((left_coords, np.full((1, left_coords.shape[1]), c)), axis=0)
        right_coords = np.concatenate((right_coords, np.full((1, right_coords.shape[1]), c)), axis=0)
        # shift coordinate based on principal point coordinate
        print(left_coords[1])
        left_coords[0] = left_coords[0] - x_c_orig
        left_coords[1] = left_coords[1] - y_c_orig
        print(left_coords[1])
        right_coords[0] = right_coords[0] - x_c_orig
        right_coords[1] = right_coords[1] - y_c_orig
        n_points = left_coords.shape[1]
        
        points_idx = 0
        points_3D_RGB = np.zeros((n_points, 6), dtype=np.float64)
        for i in range(n_points):
        # for i in range(2000000, n_points):
            print(left_coords[:, i].shape)
            print(left_coords[:, i])
            print(right_coords[:, i].shape)
            print(right_coords[:, i])
            print(Z1.shape)
            print(Z1)
            print(Z2.shape)
            print(Z2)
            print(R1)
            print(R2)
            r = R1.T @ left_coords[:, i]
            s = R2.T @ right_coords[:, i]
            b = np.array([[np.dot((Z2 - Z1), r)],
                          [np.dot((Z2 - Z1), s)]])
            A = np.array([[r.T @ r, -s.T @ r],
                          [r.T @ s, -s.T @ s]])
            params = np.linalg.solve(A, b)
            F = Z1 + params[0, 0] * r
            G = Z2 + params[1, 0] * s
            print(F)
            print(G)
            point_coords = (F + G) / 2
            print(point_coords)
            print("STOP!!!!")
            break
            points_3D_RGB[i, 0:3] = point_coords
            if points_idx % 100000 == 0:
                t = time.time()
                t_mins = (t - T0) / 60.0
                print("{} of {} points triangulated after {:.2f} mins".format(points_idx, n_points, t_mins))
            points_idx += 1
        points_3D_RGB_multi[:, :, idx] = points_3D_RGB
    points_3D_RGB_multi = points_3D_RGB_multi.mean(axis=-1)

    return points_3D_RGB_multi, aggregated_mask


def triangulate_multi_db(base_image, side_images,
                         disp_arrays, mask_arrays,
                         mappings_list, rect_params_list,
                         dbname):
    """
    Triangulate using left disparity image.

    :param base_image: file name of the base image
    :param side_images: list containing file names of the side images
    :param disp_arrays: list of left disparity image arrays
    :param mappings_list: list of dictonaries containing left and right mappings
    :param rect_params_list: list of dictionaries containing rectification parameters
    :return:
    """
    # calculate aggregated mask
    mask_list = []
    for idx, side_image in enumerate(side_images):
        K = rect_params_list[idx]["K"]
        imgparams1 = get_img_params_db(base_image, dbname)
        imgparams2 = get_img_params_db(side_image, dbname)
        R1 = imgparams1["rotation_matrix"]
        R2 = imgparams2["rotation_matrix"]
        Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
        Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
        c = K[0][0]
        # calculate image coordinates of overlap and add mask based on it
        left_extent = imgparams1["geobounds"]
        right_extent = imgparams2["geobounds"]
        mean_flying_height = get_global_param(dbname, "mean_flying_height")
        mean_z = ((imgparams1["Z"] + imgparams2["Z"]) / 2.0) - mean_flying_height
        left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
        right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
        # overlap = left_extent.intersection(right_extent)
        overlap: BaseGeometry = left_extent.intersection(right_extent)
        # If this ever isn't a Polygon, fail loudly so the assumption is explicit
        if not isinstance(overlap, Polygon):
            raise ValueError(f"Expected Polygon overlap, got {overlap.geom_type!r}")
        overlap = cast(Polygon, overlap)
        xx, yy = overlap.exterior.coords.xy
        xx = xx.tolist()
        yy = yy.tolist()
        overlap_coords_world = [(x, y, mean_z, 1) for x, y in zip(xx, yy)]
        overlap_coords_world = overlap_coords_world[:-1]
        # print(overlap_coords_world)
        t = -R1@Z1
        R_t = np.zeros((3, 4), dtype=np.float64)
        R_t[:, :-1] = R1
        R_t[:, -1] = t
        P = K@R_t
        overlap_coords_image = [P@np.array(i) for i in overlap_coords_world]
        overlap_coords_image = [i / i[-1] for i in overlap_coords_image]
        overlap_bounds = Polygon([i[:-1] for i in overlap_coords_image])
        overlap_bounds = overlap_bounds.buffer(-OVERLAP_BOUNDS_BUFFER)
        mask = rasterio.features.rasterize([overlap_bounds], out_shape=disp_arrays[idx].shape)
        mask = mask * mask_arrays[idx]
        mask_list.append(mask)

    aggregated_mask = np.ones(mask_list[0].shape)
    for i in range(len(mask_list)):
        aggregated_mask = aggregated_mask * mask_list[i]
    # plt.imshow(aggregated_mask)
    # plt.show()
    points_3D_RGB_multi = np.zeros((int((aggregated_mask==1).sum()), 6,
                                    len(side_images)), dtype=np.float64)
    for idx, side_image in enumerate(side_images):
        K = rect_params_list[idx]["K"]
        imgparams1 = get_img_params_db(base_image, dbname)
        imgparams2 = get_img_params_db(side_image, dbname)
        R1 = imgparams1["rotation_matrix"]
        R2 = imgparams2["rotation_matrix"]
        Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
        Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
        c = K[0][0]
        x_c_orig = K[0, 2]
        y_c_orig = K[1, 2]
        left_mapping = mappings_list[idx]["left"]
        right_mapping = mappings_list[idx]["right"]
        ny, nx = disp_arrays[idx].shape
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xv, yv = np.meshgrid(x, y)
        # print(xv.shape)
        # print(yv.shape)
        xv = xv[aggregated_mask==1]
        yv = yv[aggregated_mask==1]
        disp_values = disp_arrays[idx][aggregated_mask==1]
        # print(xv.shape)
        # print(yv.shape)
        third_column = np.ones(xv.shape)
        left_data = np.stack((xv, yv, third_column, disp_values))
        right_data = left_data.copy()
        # map to warped space of left image
        right_data[0:3, :] = np.linalg.inv(left_mapping) @ right_data[0:3, :]
        right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
        # add x-disparity
        right_data[0:1, :] = right_data[0:1, :] - right_data[-1:, :]
        # map to orig space of right image
        right_data[0:3, :] = right_mapping @ right_data[0:3, :]
        right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
        left_coords = left_data[0:2]
        right_coords = right_data[0:2]
        del left_data, right_data
        left_coords = np.concatenate((left_coords, np.full((1, left_coords.shape[1]), c)), axis=0)
        right_coords = np.concatenate((right_coords, np.full((1, right_coords.shape[1]), c)), axis=0)
        # shift coordinate based on principal point coordinate
        left_coords[0] = left_coords[0] - x_c_orig
        left_coords[1] = left_coords[1] - y_c_orig
        right_coords[0] = right_coords[0] - x_c_orig
        right_coords[1] = right_coords[1] - y_c_orig
        n_points = left_coords.shape[1]
        
        # points_idx = 0
        # points_3D_RGB = np.zeros((n_points, 6), dtype=np.float64)
        # for i in range(n_points):
        #     r = R1.T @ left_coords[:, i]
        #     s = R2.T @ right_coords[:, i]
        #     b = np.array([[np.dot((Z2 - Z1), r)],
        #                   [np.dot((Z2 - Z1), s)]])
        #     A = np.array([[r.T @ r, -s.T @ r],
        #                   [r.T @ s, -s.T @ s]])
        #     params = np.linalg.solve(A, b)
        #     F = Z1 + params[0, 0] * r
        #     G = Z2 + params[1, 0] * s
        #     point_coords = (F + G) / 2
        #     points_3D_RGB[i, 0:3] = point_coords
        #     if points_idx % 100000 == 0:
        #         t = time.time()
        #         t_mins = (t - T0) / 60.0
        #         print("{} of {} points triangulated after {:.2f} mins".format(points_idx, n_points, t_mins))
        #     points_idx += 1
        # points_3D_RGB_multi[:, :, idx] = points_3D_RGB
        from joblib import Parallel, delayed
        def process_point(i, n_points, R1, R2, left_coords, right_coords, Z1, Z2):
            r = R1.T @ left_coords[:, i]
            s = R2.T @ right_coords[:, i]
            b = np.array([[np.dot((Z2 - Z1), r)],
                          [np.dot((Z2 - Z1), s)]])
            A = np.array([[r.T @ r, -s.T @ r],
                          [r.T @ s, -s.T @ s]])
            params = np.linalg.solve(A, b)
            F = Z1 + params[0, 0] * r
            G = Z2 + params[1, 0] * s
            point_coords = (F + G) / 2
            return i, point_coords

        def parallel_process(n_points, R1, R2, left_coords, right_coords, Z1, Z2):
            points_3D_RGB = np.zeros((n_points, 6), dtype=np.float64)
            results = Parallel(n_jobs=-1)(delayed(process_point)(i, n_points, R1, R2, left_coords, right_coords, Z1, Z2) for i in range(n_points))
            for i, point_coords in results:
                points_3D_RGB[i, 0:3] = point_coords
            return points_3D_RGB

        result = parallel_process(n_points, R1, R2, left_coords, right_coords, Z1, Z2)
        points_3D_RGB_multi[:, :, idx] = result
    points_3D_RGB_multi = points_3D_RGB_multi.mean(axis=-1)

    return points_3D_RGB_multi, aggregated_mask


# def left_disparity_to_pointcloud(img1, img2,
#                                  disp_array1, orig_disp_array1,
#                                  rect_params, img_params):
#     disp_array1 = disp_array1.copy()
#     H1 = rect_params["H1"]
#     H2 = rect_params["H2"]
#     H_shift = rect_params["H_shift"]
#     K = rect_params["K"]
#     R = rect_params["R"]
#     imgparams1 = img_params[img1]
#     imgparams2 = img_params[img2]
#     R1 = imgparams1["rotation_matrix"]
#     R2 = imgparams2["rotation_matrix"]
#     Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
#     Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
#     baseline = np.linalg.norm(Z1-Z2)
#     f_x = K[0][0]
#     f_y = K[1][1]
#     p_x = K[0][-1]
#     p_y = K[1][-1]
#     print(baseline)
#     # calculate image coordinates of overlap
#     left_extent = imgparams1["geobounds"]
#     right_extent = imgparams2["geobounds"]
#     mean_z = imgparams1["mean_gcp_height"]
#     left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
#     right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
#     overlap = left_extent.intersection(right_extent)
#     xx, yy = overlap.exterior.coords.xy
#     xx = xx.tolist()
#     yy = yy.tolist()
#     overlap_coords_world = [(x, y, mean_z, 1) for x, y in zip(xx, yy)]
#     overlap_coords_world = overlap_coords_world[:-1]
#     # print(overlap_coords_world)
#     t = -R1@Z1
#     R_t = np.zeros((3, 4), dtype=np.float64)
#     R_t[:, :-1] = R1
#     R_t[:, -1] = t
#     P = K@R_t
#     overlap_coords_image = [P@np.array(i) for i in overlap_coords_world]
#     overlap_coords_image = [i / i[-1] for i in overlap_coords_image]
#     overlap_bounds = Polygon([i[:-1] for i in overlap_coords_image])
#     overlap_bounds = overlap_bounds.buffer(-OVERLAP_BOUNDS_BUFFER)
#     mask = rasterio.features.rasterize([overlap_bounds], out_shape=orig_disp_array1.shape)
#     # ======================================================
#     ny, nx = orig_disp_array1.shape
#     x = np.linspace(0, nx - 1, nx)
#     y = np.linspace(0, ny - 1, ny)
#     xv, yv = np.meshgrid(x, y)
#     xv = xv[mask==1]
#     yv = yv[mask==1]
#     # plt.imshow(orig_disp_array1)
#     # plt.show()
#     disp_values = orig_disp_array1[mask==1]
#     # disp_copy = orig_disp_array1.copy()
#     # disp_copy[mask!=1] = np.NaN
#     # disp_copy = f_x * baseline / disp_copy
#     # plt.imshow(disp_copy)
#     # plt.show()
#     z_camera = f_x * baseline / disp_values
#     x_camera = (xv - p_x) * (z_camera / f_x)
#     y_camera = (yv - p_y) * (z_camera / f_y)
#     coords_camera = np.stack([x_camera, y_camera, z_camera, np.ones(z_camera.size)], axis=1)
#     T1 = np.zeros((4, 4), dtype=np.float64)
#     T1[:3, :3] = R
#     T1[-1, -1] = 1
#     T1[:-1, -1] = Z1
#     print(coords_camera)
#     print(T1)
#     coords_world = coords_camera @ np.linalg.inv(T1).T
#     print(coords_world)
#     coords_world = coords_world / coords_world[:, -1:]
#     print(coords_world)
#     # coords_world = Z1 + coords_world
#     # print(coords_world)
#     points_3D = np.zeros((coords_world.shape[0], 6), dtype=np.float64)
#     points_3D[:, 0:3] = coords_world[:, 0:3]
#     # ======================================================
#
#     # ======================================================
#     # overlap_coords_warped = []
#     # for x, y in zip(overlap_bounds.exterior.xy[0][:-1],
#     #                 overlap_bounds.exterior.xy[1][:-1]):
#     #     bounds_corner_orig = np.array([x, y, 1], dtype=np.float64)
#     #     bounds_corner_warped = np.linalg.inv(H1 @ H_shift) @ bounds_corner_orig
#     #     bounds_corner_warped = bounds_corner_warped / bounds_corner_warped[-1]
#     #     overlap_coords_warped.append(bounds_corner_warped)
#     # overlap_bounds_warped = Polygon([i[:-1] for i in overlap_coords_warped])
#     # mask_warped = rasterio.features.rasterize([overlap_bounds_warped], out_shape=disp_array1.shape)
#     # ny, nx = disp_array1.shape
#     # x = np.linspace(0, nx - 1, nx)
#     # y = np.linspace(0, ny - 1, ny)
#     # xv, yv = np.meshgrid(x, y)
#     # xv = xv[mask_warped==1]
#     # yv = yv[mask_warped==1]
#     # disp_values = disp_array1[mask_warped==1]
#     # # plt.imshow(disp_array1)
#     # # plt.show()
#     # # disp_array1[mask_warped!=1] = np.NaN
#     # # plt.imshow(disp_array1)
#     # # plt.show()
#     # p_coords_warped = np.linalg.inv(H1 @ H_shift) @ np.array([p_x, p_y, 1], dtype=np.float64)
#     # p_coords_warped = p_coords_warped / p_coords_warped[-1]
#     # p_x_warped = p_coords_warped[0]
#     # p_y_warped = p_coords_warped[1]
#     # z_camera = f_x * baseline / disp_values
#     # x_camera = (xv - p_x_warped) * (z_camera / f_x)
#     # y_camera = (yv - p_y_warped) * (z_camera / f_y)
#     # coords_camera = np.stack([x_camera, y_camera, z_camera, np.ones(z_camera.size)], axis=1)
#     # T1 = np.zeros((4, 4), dtype=np.float64)
#     # T1[:3, :3] = R1
#     # T1[-1, -1] = 1
#     # T1[:-1, -1] = Z1
#     # coords_world = coords_camera @ T1.T
#     # coords_world = coords_world / coords_world[:, -1:]
#     # print(coords_world)
#     # # coords_world = Z1 + coords_world
#     # # print(coords_world)
#     # points_3D = np.zeros((coords_world.shape[0], 6), dtype=np.float64)
#     # points_3D[:, 0:3] = coords_world[:, 0:3]
#     # ======================================================
#
#     # overlap_coords_warped = []
#     # for x, y in zip(overlap_bounds.exterior.xy[0][:-1],
#     #                 overlap_bounds.exterior.xy[1][:-1]):
#     #     bounds_corner_orig = np.array([x, y, 1], dtype=np.float64)
#     #     bounds_corner_warped = np.linalg.inv(H1 @ H_shift) @ bounds_corner_orig
#     #     bounds_corner_warped = bounds_corner_warped / bounds_corner_warped[-1]
#     #     overlap_coords_warped.append(bounds_corner_warped)
#     # overlap_bounds_warped = Polygon([i[:-1] for i in overlap_coords_warped])
#     # mask_warped = rasterio.features.rasterize([overlap_bounds_warped], out_shape=disp_array1.shape)
#     # ny_warped, nx_warped = disp_array1.shape
#     # x_warped = np.linspace(0, nx_warped - 1, nx_warped)
#     # y_warped = np.linspace(0, ny_warped - 1, ny_warped)
#     # xv_warped, yv_warped = np.meshgrid(x_warped, y_warped)
#     # xv_warped = xv_warped[mask_warped==1]
#     # yv_warped = yv_warped[mask_warped==1]
#     # disp_values_warped = disp_array1[mask_warped==1]
#     # left_coords_warped = np.stack([xv_warped, yv_warped, np.ones(xv_warped.size)], axis=1)
#     # left_coords_orig = left_coords_warped @ (H1 @ H_shift).T
#     # left_coords_orig = left_coords_orig / left_coords_orig[:, -1:]
#     # left_coords_orig = left_coords_orig[:, :-1]
#     # right_coords_warped = np.stack([xv_warped + disp_values_warped, yv_warped, np.ones(xv_warped.size)], axis=1)
#     # right_coords_orig = right_coords_warped @ (H2 @ H_shift).T
#     # right_coords_orig = right_coords_orig / right_coords_orig[:, -1:]
#     # right_coords_orig = right_coords_orig[:, :-1]
#     # disp_values_orig = ((right_coords_orig[:, 0] - left_coords_orig[:, 0]) ** 2 + \
#     #                     (right_coords_orig[:, 1] - left_coords_orig[:, 1]) ** 2) ** 0.5
#     # print(disp_values_warped)
#     # plt.plot(disp_values_warped)
#     # plt.show()
#     # print(disp_values_orig)
#     # plt.plot(disp_values_orig)
#     # plt.show()
#     # p_x_orig = p_x
#     # p_y_orig = p_y
#     # xv_orig = left_coords_orig[:, 0]
#     # yv_orig = left_coords_orig[:, 1]
#     # # p_coords_warped = np.linalg.inv(H1 @ H_shift) @ np.array([p_x, p_y, 1], dtype=np.float64)
#     # # p_coords_warped = p_coords_warped / p_coords_warped[-1]
#     # # p_x_warped = p_coords_warped[0]
#     # # p_y_warped = p_coords_warped[1]
#     # z_camera = f_x * baseline / disp_values_orig
#     # x_camera = (xv_orig - p_x_orig) * (z_camera / f_x)
#     # y_camera = (yv_orig - p_y_orig) * (z_camera / f_y)
#     # coords_camera = np.stack([x_camera, y_camera, z_camera, np.ones(z_camera.size)], axis=1)
#     # T1 = np.zeros((4, 4), dtype=np.float64)
#     # T1[:3, :3] = R1
#     # T1[-1, -1] = 1
#     # T1[:-1, -1] = Z1
#     # coords_world = coords_camera @ T1.T
#     # coords_world = coords_world / coords_world[:, -1:]
#     # print(coords_world)
#     # points_3D = np.zeros((coords_world.shape[0], 6), dtype=np.float64)
#     # points_3D[:, 0:3] = coords_world[:, 0:3]
#     return points_3D, mask


def write_ply(vertices, colors, filename, pair_idx):
    # save point cloud as ply
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])
    if pair_idx == 0:
        ply_header = '''ply
            format ascii 1.0
            element vertex %(vert_num)d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            end_header\n'''
        with open(filename, "w") as f:
            f.write(ply_header % dict(vert_num=len(vertices)))
            np.savetxt(f, vertices, "%f %f %f %d %d %d")
    else:
        with open(filename, "r") as f:
            content = f.read()
            r = re.findall("vertex \d+", content)
            current_count = int(r[0][7:])
            new_count = current_count + len(vertices)
            content_new = re.sub("vertex \d+", "vertex {}".format(new_count), content)
        with open(filename, "w") as f:
            f.write(content_new)
            np.savetxt(f, vertices, "%f %f %f %d %d %d")


def write_ply_fpc_format(dbname, vertices, colors, imgname, num_images):
    # save point cloud as ply
    filename = os.path.join(TMP_PLY, "{}.ply".format(imgname[:-4]))
    colors = colors.reshape(-1, 3)
    vertices = vertices.reshape(-1, 3)
    nviews = np.full((len(vertices), 1), fill_value=num_images)
    x_shift = get_global_param(dbname, "x_shift")
    y_shift = get_global_param(dbname, "y_shift")
    vertices[:, 0] -= x_shift
    vertices[:, 1] -= y_shift
    vertices = np.hstack([vertices, colors, nviews])
    ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar diffuse_red
property uchar diffuse_green
property uchar diffuse_blue
property uchar views
end_header\n'''
    with open(filename, "w") as f:
        f.write(ply_header % dict(vert_num=len(vertices)))
        np.savetxt(f, vertices, "%f %f %f %d %d %d %d")


def subset_image(imgarray, sub_height, sub_width,
                 sub_overlap_height, sub_overlap_width):
    if len(imgarray.shape) == 2:
        nrows, ncols = imgarray.shape
        nbands = 1
        imgarray = np.expand_dims(imgarray, axis=-1)
    elif len(imgarray.shape) == 3:
        nrows, ncols, nbands = imgarray.shape
    img_subsets = np.empty((0, sub_height, sub_width, nbands), dtype=np.float64)
    startrow = 0
    endrow = sub_height
    while startrow < nrows:
        startcol = 0
        endcol = sub_width
        while startcol < ncols:
            if (endrow <= nrows) and (endcol <= ncols):
                subset = imgarray[startrow:endrow, startcol:endcol, :]
            elif (endrow > nrows) and (endcol <= ncols):
                subset = imgarray[-sub_height:, startcol:endcol, :]
            elif (endrow > nrows) and (endcol > ncols):
                subset = imgarray[-sub_height:, -sub_width:, :]
            elif (endrow <= nrows) and (endcol > ncols):
                subset = imgarray[startrow:endrow, -sub_width:, :]
            # plt.imshow(subset)
            # plt.show()
            subset = np.expand_dims(subset, axis=0)
            img_subsets = np.concatenate((img_subsets, subset), axis=0)
            startcol = endcol - sub_overlap_width
            endcol = startcol + sub_width
        startrow = endrow - sub_overlap_height
        endrow = startrow + sub_height
    return img_subsets

def calc_disparity(rect_img1, rect_img2, disparity_method):
    imgpath1 = os.path.join(TMP_IMG_LEFT, "{}.tif".format(TMP_FNAME))
    imgpath2 = os.path.join(TMP_IMG_RIGHT, "{}.tif".format(TMP_FNAME))
    write_array(rect_img1, imgpath1)
    write_array(rect_img2, imgpath2)


    if not os.path.exists(TMP_RECT_FNAME_LIST):
        with open(TMP_RECT_FNAME_LIST, "w") as f:
            f.write("img/left/{0}.tif img/right/{0}.tif".format(TMP_FNAME))


    rect_height, rect_width, _ = rect_img1.shape
    if rect_height % 128 == 0:
        height_snap = rect_height
    else:
        height_snap = 128 * ((rect_height // 128) + 1)
    if rect_width % 128 == 0:
        width_snap = rect_width
    else:
        width_snap = 128 * ((rect_width // 128) + 1)
    if disparity_method ==  "CNN":
        # **TODO: pass global constants to function**
        write_disparity_PASMNet(height_snap, width_snap, TMP_RECT_FNAME_LIST, TMP_ROOT, MODEL_PATH)

        disp_img1 = gdal.Open(TMP_LEFT_DISP)
        disp_array1 = disp_img1.ReadAsArray()
        # use parallax-attention maps based mask
        tmp_mask_fname = TMP_LEFT_DISP[:-4] + "_mask.tif"
        mask_array1 = gdal.Open(tmp_mask_fname)
        mask_array1 = mask_array1.ReadAsArray()
        mask_array1 = transform.resize(mask_array1, disp_array1.shape, anti_aliasing=False)


        # print(mask_array1.shape)
        # print(disp_array1.shape)
        # disp_array1 = (disp_array1 / (2.0 ** 15)) * 128
        # disp_array1 = disp_array1 / 256.0
        # disp_array1 = disp_array1 / 512.0
        # plt.imshow(disp_array1)
        # plt.show()
    elif disparity_method == "SGM":
        write_disparity_SGM(imgpath1, imgpath2)

        disp_img1 = gdal.Open(SGM_DISP)
        disp_array1 = disp_img1.ReadAsArray()
        disp_array1 = - disp_array1
        mask_array1 = gdal.Open(SGM_MASK)
        mask_array1 = mask_array1.ReadAsArray()
        mask_array1[mask_array1==0] = 1
        mask_array1[mask_array1!=1] = 0
    return np.dstack((disp_array1, mask_array1))


def stitch_disparity(rect_img1, img1_subsets, img2_subsets, disparity_method):
    nrows, ncols, nbands = rect_img1.shape
    disp_array = np.zeros(shape=(nrows, ncols, 2), dtype=np.float64)
    n_subsets, sub_height, sub_width, _ = img1_subsets.shape
    sub_overlap_width = SUBSET_WIDTH_OVERLAP
    sub_overlap_height = SUBSET_HEIGHT_OVERLAP
    t_subset_0 = time.time()
    subset_idx = 0
    startrow = 0
    endrow = sub_height
    while startrow < nrows:
        startcol = 0
        endcol = sub_width
        while startcol < ncols:
            img1_subset = img1_subsets[subset_idx]
            img2_subset = img2_subsets[subset_idx]
            # print(nrows, ncols)
            # print(startrow, endrow)
            # print(startcol, endcol)
            disp_array_subset = calc_disparity(img1_subset, img2_subset, disparity_method)
            # fig, ax = plt.subplots(nrows=1, ncols=3)
            # ax[0].imshow(img1_subset)
            # ax[1].imshow(img2_subset)
            # ax[2].imshow(disp_array_subset)
            # plt.show()
            if (endrow <= nrows) and (endcol <= ncols):
                fill_most_disparity(disp_array, disp_array_subset,
                                    startrow, endrow, startcol, endcol,
                                    sub_height, sub_width,
                                    sub_overlap_height, sub_overlap_width,
                                    nrows, ncols)
            elif (endrow > nrows) and (endcol <= ncols):
                fill_most_disparity(disp_array, disp_array_subset,
                                    nrows - sub_height, nrows, startcol, endcol,
                                    sub_height, sub_width,
                                    sub_overlap_height, sub_overlap_width,
                                    nrows, ncols)
            elif (endrow > nrows) and (endcol > ncols):
                fill_most_disparity(disp_array, disp_array_subset,
                                    nrows - sub_height, nrows, ncols - sub_width, ncols,
                                    sub_height, sub_width,
                                    sub_overlap_height, sub_overlap_width,
                                    nrows, ncols)
            elif (endrow <= nrows) and (endcol > ncols):
                fill_most_disparity(disp_array, disp_array_subset,
                                    startrow, endrow, ncols - sub_width, ncols,
                                    sub_height, sub_width,
                                    sub_overlap_height, sub_overlap_width,
                                    nrows, ncols)
            startcol = endcol - sub_overlap_width
            endcol = startcol + sub_width
            subset_idx += 1
        startrow = endrow - sub_overlap_height
        endrow = startrow + sub_height
        t_subset_mins = (time.time() - t_subset_0) / 60.0
        print("finished processing {} of {} subsets in {:.2f} mins".format(subset_idx, n_subsets, t_subset_mins))
    return disp_array


def fill_most_disparity(disp_array, disp_array_subset,
                        startrow, endrow, startcol, endcol,
                        sub_height, sub_width,
                        sub_overlap_height, sub_overlap_width,
                        nrows, ncols):
    if (startrow == 0) and (startcol == 0):
        disp_array[startrow:endrow - (sub_overlap_height // 2),
        startcol:endcol - (sub_overlap_width // 2)] = disp_array_subset[:-(sub_overlap_height // 2),
                                                                        :-(sub_overlap_width // 2)]
    elif (startrow == 0) and (0 < startcol < (ncols - sub_width)):
        disp_array[startrow:endrow - (sub_overlap_height // 2),
        startcol + (sub_overlap_width // 2):endcol - (sub_overlap_width // 2)] = disp_array_subset[:-(sub_overlap_height // 2),
                                                                     (sub_overlap_width // 2):-(sub_overlap_width // 2)]
    elif (startrow == 0) and (startcol >= (ncols - sub_width)):
        disp_array[startrow:endrow - (sub_overlap_height // 2),
        startcol + (sub_overlap_width // 2):] = disp_array_subset[:-(sub_overlap_height // 2),
                                          (sub_overlap_width // 2):]
    elif (0 < startrow < (nrows - sub_height)) and (startcol == 0):
        disp_array[startrow + (sub_overlap_height // 2):endrow - (sub_overlap_height // 2),
        startcol:endcol - (sub_overlap_width // 2)] = disp_array_subset[(sub_overlap_height // 2):-(sub_overlap_height // 2),
                                                :-(sub_overlap_width // 2)]
    elif (0 < startrow < (nrows - sub_height)) and (0 < startcol < (ncols - sub_width)):
        disp_array[startrow + (sub_overlap_height // 2):endrow - (sub_overlap_height // 2),
        startcol + (sub_overlap_width // 2):endcol - (sub_overlap_width // 2)] = disp_array_subset[
                                                                     (sub_overlap_height // 2):-(sub_overlap_height // 2),
                                                                     (sub_overlap_width // 2):-(sub_overlap_width // 2)]
    elif (0 < startrow < (nrows - sub_height)) and (startcol >= (ncols - sub_width)):
        disp_array[startrow + (sub_overlap_height // 2):endrow - (sub_overlap_height // 2),
        startcol + (sub_overlap_width // 2):] = disp_array_subset[(sub_overlap_height // 2):-(sub_overlap_height // 2),
                                          (sub_overlap_width // 2):]
    elif (startrow >= (nrows - sub_height)) and (startcol == 0):
        disp_array[startrow + (sub_overlap_height // 2):,
        startcol:endcol - (sub_overlap_width // 2)] = disp_array_subset[(sub_overlap_height // 2):,
                                                :-(sub_overlap_width // 2)]
    elif (startrow >= (nrows - sub_height)) and (0 < startcol < (ncols - sub_width)):
        disp_array[startrow + (sub_overlap_height // 2):,
        startcol + (sub_overlap_width // 2):endcol - (sub_overlap_width // 2)] = disp_array_subset[(sub_overlap_height // 2):,
                                                                     (sub_overlap_width // 2):-(sub_overlap_width // 2)]
    elif (startrow >= (nrows - sub_height)) and (startcol >= (ncols - sub_width)):
        disp_array[startrow + (sub_overlap_height // 2):,
        startcol + (sub_overlap_width // 2):] = disp_array_subset[(sub_overlap_height // 2):,
                                          (sub_overlap_width // 2):]


def calc_disp_reduction(left_img_params, right_img_params, rect_params):
    # z_world = left_img_params["mean_gcp_height"]
    z_f = left_img_params["mean_flying_height"]
    K_left = left_img_params["camera_matrix"]
    R_left = left_img_params["rotation_matrix"]
    C_left = np.array([[left_img_params["X"]],
                       [left_img_params["Y"]],
                       [left_img_params["Z"]]], dtype=np.float64)
    K_right = right_img_params["camera_matrix"]
    R_right = right_img_params["rotation_matrix"]
    C_right = np.array([[right_img_params["X"]],
                        [right_img_params["Y"]],
                        [right_img_params["Z"]]], dtype=np.float64)
    P_left = calc_P_matrix(left_img_params)
    P_right = calc_P_matrix(right_img_params)
    H1 = rect_params["H1"]
    H2 = rect_params["H2"]
    H_shift = rect_params["H_shift"]
    # use principal point
    x_img_left = 0
    y_img_left = 0
    img_coords_left = np.array([[K_left[0, 2]],
                                [K_left[1, 2]],
                                [1]], dtype=np.float64)
    # get rectified image coordinates
    warped_coords_left = np.linalg.inv(H1 @ H_shift) @ img_coords_left
    warped_coords_left = warped_coords_left / warped_coords_left[-1]
    # print(warped_coords_left)
    # get estimated world coordinates
    f_x = K_left[0][0]
    f_y = K_left[1][1]
    coords_delta = np.array([[(x_img_left * z_f / f_x)],
                             [(y_img_left * z_f / f_y)],
                             [z_f]], dtype=np.float64)
    coords_world = np.linalg.inv(R_left) @ coords_delta + C_left
    coords_world_h = np.ones(shape=(coords_world.shape[0] + 1,
                                    coords_world.shape[1]), dtype=np.float64)
    coords_world_h[:-1, :] = coords_world
    img_coords_right = P_right @ coords_world_h
    img_coords_right = img_coords_right / img_coords_right[-1, :]
    warped_coords_right = np.linalg.inv(H2 @ H_shift) @ img_coords_right
    warped_coords_right = warped_coords_right / warped_coords_right[-1, :]
    # print(warped_coords_right)
    disp_shift = warped_coords_left[0, 0] - warped_coords_right[0, 0]
    # print(disp_shift)
    return int(disp_shift * DISPARITY_SHIFT_RATIO)


def calc_disp_reduction_db(left_img_params,
                           right_img_params,
                           rect_params,
                           dbname):
    # z_world = left_img_params["mean_gcp_height"]
    # z_f = left_img_params["mean_flying_height"]
    z_f = get_global_param(dbname, "mean_flying_height")
    K_left = left_img_params["camera_matrix"]
    R_left = left_img_params["rotation_matrix"]
    C_left = np.array([[left_img_params["X"]],
                       [left_img_params["Y"]],
                       [left_img_params["Z"]]], dtype=np.float64)
    K_right = right_img_params["camera_matrix"]
    R_right = right_img_params["rotation_matrix"]
    C_right = np.array([[right_img_params["X"]],
                        [right_img_params["Y"]],
                        [right_img_params["Z"]]], dtype=np.float64)
    P_left = calc_P_matrix(left_img_params)
    P_right = calc_P_matrix(right_img_params)
    H1 = rect_params["H1"]
    H2 = rect_params["H2"]
    H_shift = rect_params["H_shift"]
    # use principal point
    x_img_left = 0
    y_img_left = 0
    img_coords_left = np.array([[K_left[0, 2]],
                                [K_left[1, 2]],
                                [1]], dtype=np.float64)
    # get rectified image coordinates
    warped_coords_left = np.linalg.inv(H1 @ H_shift) @ img_coords_left
    warped_coords_left = warped_coords_left / warped_coords_left[-1]
    # print(warped_coords_left)
    # get estimated world coordinates
    f_x = K_left[0][0]
    f_y = K_left[1][1]
    coords_delta = np.array([[(x_img_left * z_f / f_x)],
                             [(y_img_left * z_f / f_y)],
                             [z_f]], dtype=np.float64)
    coords_world = np.linalg.inv(R_left) @ coords_delta + C_left
    coords_world_h = np.ones(shape=(coords_world.shape[0] + 1,
                                    coords_world.shape[1]), dtype=np.float64)
    coords_world_h[:-1, :] = coords_world
    img_coords_right = P_right @ coords_world_h
    img_coords_right = img_coords_right / img_coords_right[-1, :]
    warped_coords_right = np.linalg.inv(H2 @ H_shift) @ img_coords_right
    warped_coords_right = warped_coords_right / warped_coords_right[-1, :]
    # print(warped_coords_right)
    disp_shift = warped_coords_left[0, 0] - warped_coords_right[0, 0]
    # print(disp_shift)
    return int(disp_shift * DISPARITY_SHIFT_RATIO)


def shift_right_image(rect_img, disparity_shift):
    shift_matrix = np.array([[1, 0, disparity_shift],
                             [0, 1, 0],
                             [0, 0, 1 ]])
    tform = transform.ProjectiveTransform(matrix=shift_matrix)
    tf_img = transform.warp(rect_img, tform, output_shape=rect_img.shape)
    return tf_img


def get_left_disparity(LEFT, RIGHT, img_params, DF, disparity_method):
    # extract left and right image dictionary objects
    LEFT_IMG = LEFT["fname"]
    LEFT_IMG_OBJECT = LEFT["object"]
    LEFT_IMG_ARRAY = LEFT["array"]
    RIGHT_IMG = RIGHT["fname"]
    # TODO: remove useless? *img_objects*
    RIGHT_IMG_OBJECT = RIGHT["object"]
    RIGHT_IMG_ARRAY = RIGHT["array"]
    LEFT_IMG_PARAMS = img_params[LEFT_IMG]
    RIGHT_IMG_PARAMS = img_params[RIGHT_IMG]

    # rectify image pairs
    print("rectifying {} and {} image pair...".format(LEFT_IMG, RIGHT_IMG))
    img_rect_params_dict = rectify_stereopair(LEFT_IMG_ARRAY, RIGHT_IMG_ARRAY,
                                              LEFT_IMG_PARAMS, RIGHT_IMG_PARAMS,
                                              DF)
    rect_img1, rect_img2 = img_rect_params_dict["image_pairs"]
    rect_params = img_rect_params_dict["rect_params"]
    # import pickle
    # with open('{}_{}_rect.params'.format(LEFT_IMG, RIGHT_IMG), 'wb') as f:
    #     pickle.dump(rect_params, f)
    # print("pickled rect_params!!!")
    # orig_img1, orig_img2 = LEFT_IMG_OBJECT, RIGHT_IMG_OBJECT
    # orig_imgarray1 = LEFT_IMG_ARRAY
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # reduce disparities so minimum is closer to zero
    print("reducing absolute disparity values...")
    disparity_shift = calc_disp_reduction(LEFT_IMG_PARAMS, RIGHT_IMG_PARAMS, rect_params)
    shifted_rect_img2 = shift_right_image(rect_img2, -disparity_shift)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # tile images before disparity estimation so they fit to memory
    print("subsetting image pairs...")
    img1_subsets = subset_image(rect_img1, SUBSET_HEIGHT, SUBSET_WIDTH,
                                SUBSET_HEIGHT_OVERLAP, SUBSET_WIDTH_OVERLAP)
    img2_subsets = subset_image(shifted_rect_img2, SUBSET_HEIGHT, SUBSET_WIDTH,
                                SUBSET_HEIGHT_OVERLAP, SUBSET_WIDTH_OVERLAP)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # calculate disparity and mask maps
    print("calculating one left disparity image...")
    disp_arrays = stitch_disparity(rect_img1, img1_subsets, img2_subsets, disparity_method)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # shift disparity array back to actual disparities
    disp_arrays[:, :, 0] = disp_arrays[:, :, 0] + disparity_shift
    return disp_arrays, rect_params


def get_left_disparity_db(left, right, DF, disparity_method, dbname):
    # extract left and right image dictionary objects
    left_img = left["fname"]
    left_img_array = left["array"]
    right_img = right["fname"]
    right_img_array = right["array"]
    left_img_params = get_img_params_db(left_img, dbname)
    right_img_params = get_img_params_db(right_img, dbname)
    # print(left_img)
    # print(right_img)
    # print(left_img_params)
    # print(right_img_params)
    # rectify image pairs
    print("rectifying {} and {} image pair...".format(left_img, right_img))
    img_rect_params_dict = rectify_stereopair(left_img_array, right_img_array,
                                              left_img_params, right_img_params,
                                              DF)
    rect_img1, rect_img2 = img_rect_params_dict["image_pairs"]
    rect_img1_fname = left["fname"][:-4] + right["fname"][:-4] + "_rectified" + ".tif"
    rect_img1_path = os.path.join(TMP_DISP_WARP, rect_img1_fname)
    write_array_float(rect_img1, rect_img1_path)
    rect_img2_fname = right["fname"][:-4] + left["fname"][:-4] + "_rectified" + ".tif"
    rect_img2_path = os.path.join(TMP_DISP_WARP, rect_img2_fname)
    write_array_float(rect_img2, rect_img2_path)
    rect_params = img_rect_params_dict["rect_params"]
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # reduce disparities so minimum is closer to zero
    print("reducing absolute disparity values...")
    disparity_shift = calc_disp_reduction_db(left_img_params, right_img_params, rect_params, dbname)
    shifted_rect_img2 = shift_right_image(rect_img2, -disparity_shift)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # tile images before disparity estimation so they fit to memory
    # **TODO: PASS OFFSETS TO THE FUNCTION**
    print("subsetting image pairs...")
    img1_subsets = subset_image(rect_img1, SUBSET_HEIGHT, SUBSET_WIDTH,
                                SUBSET_HEIGHT_OVERLAP, SUBSET_WIDTH_OVERLAP)
    img2_subsets = subset_image(shifted_rect_img2, SUBSET_HEIGHT, SUBSET_WIDTH,
                                SUBSET_HEIGHT_OVERLAP, SUBSET_WIDTH_OVERLAP)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # calculate disparity and mask maps
    print("calculating one left disparity image...")
    disp_arrays = stitch_disparity(rect_img1, img1_subsets, img2_subsets, disparity_method)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    # shift disparity array back to actual disparities
    disp_arrays[:, :, 0] = disp_arrays[:, :, 0] + disparity_shift
    return disp_arrays, rect_params


def compare_disparity_maps(LRdisparray, RLdisparray):
    ny, nx = LRdisparray.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    disp_values = LRdisparray.flatten()

    left_disparities = np.stack((xv.flatten(), yv.flatten(), disp_values))
    right_disparities = left_disparities.copy()
    right_disparities[0:1, :] = right_disparities[0:1, :] - right_disparities[-1:, :]
    for i in range(right_disparities.shape[1]):
        right_disparities[-1:, i] = RLdisparray[int(right_disparities[1, i]), int(right_disparities[0, i])]

    deviations = left_disparities[-1, :] - right_disparities[-1, :]
    deviations = np.reshape(deviations, LRdisparray.shape)

    return deviations


def init_DB(dbname,
            DATAROOT, IMGDIR, LAS_FNAME,
            DS_factor, T0, DF, epsg):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="postgres")

    # Create a cursor to perform database operations
    cur = conn.cursor()
    conn.autocommit = True
    # Print PostgreSQL details
    Q_CHECK_DB_EXIST = """SELECT
                          EXISTS(
                            SELECT datname
                            FROM pg_catalog.pg_database
                            WHERE lower(datname) = lower('{}'));""".format(dbname)
    cur.execute(Q_CHECK_DB_EXIST)
    db_created = cur.fetchone()[0]
    if db_created:
        print("point cloud DB already initialized")
        cur.close()
        conn.close()
    else:
        Q_CREATE_PDB = """CREATE DATABASE {};""".format(dbname)
        Q_GRANT_PRIV = """GRANT ALL PRIVILEGES ON DATABASE {} TO postgres""".format(dbname)
        cur.execute(Q_CREATE_PDB)
        cur.execute(Q_GRANT_PRIV)
        cur.close()
        conn.close()
        create_db_extensions(dbname)
        create_db_tables(dbname)
        metric_calib = os.path.join(DATAROOT, "image_orientations.xyz")
        img_params = init_img_params(metric_calib, IMGDIR, LAS_FNAME, DS_factor, T0)
        estimated_GSD = estimate_GSD(img_params, DF)
        # print(img_params["2021-04-23_13-17-22_S2223314_DxO.jpg"])
        # print(img_params["2021-04-23_13-17-20_S2223314_DxO.jpg"])
        populate_imgparams_table(img_params, estimated_GSD, dbname)
        mean_flying_height = img_params[list(img_params.keys())[0]]["mean_flying_height"]
        img_Xs = [img_params[i]["X"] for i, _ in img_params.items()]
        img_Ys = [img_params[i]["Y"] for i, _ in img_params.items()]
        X_shift = np.array(img_Xs).min()
        Y_shift = np.array(img_Ys).min()
        populate_globals_table(dbname,
                               estimated_GSD,
                               mean_flying_height,
                               epsg,
                               X_shift,
                               Y_shift)
        print("point cloud DB initialized")


def populate_globals_table(dbname,
                           estimated_GSD,
                           mean_flying_height,
                           epsg,
                           X_shift,
                           Y_shift):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database=dbname)

    # Create a cursor to perform database operations
    cur = conn.cursor()
    conn.autocommit = True
    Q_INSERT_GLOBALS = """INSERT INTO globals
                            (estimated_GSD,
                             mean_flying_height,
                             epsg,
                             X_shift,
                             Y_shift)
                          VALUES
                            ({estimated_GSD},
                             {mean_flying_height},
                             {epsg},
                             {X_shift},
                             {Y_shift});"""
    cur.execute(Q_INSERT_GLOBALS.format(estimated_GSD=estimated_GSD,
                                        mean_flying_height=mean_flying_height,
                                        epsg=epsg,
                                        X_shift=X_shift,
                                        Y_shift=Y_shift))


def create_db_extensions(dbname):
    try:
        # Connect to an existing database
        conn2 = psycopg2.connect(user="postgres",
                                 password="postgres",
                                 host="127.0.0.1",
                                 port="5432",
                                 database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur2 = conn2.cursor()
        conn2.autocommit = True
        # Print PostgreSQL details
        # Q_ENABLE_EXTENSIONS = """CREATE EXTENSION postgis;
        #                          CREATE EXTENSION pointcloud;
        #                          CREATE EXTENSION pointcloud_postgis;
        #                          CREATE EXTENSION plpython3u;"""
        # print("db spatial packages enabled")
        # cur2.execute(Q_ENABLE_EXTENSIONS)
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn2):
            cur2.close()
            conn2.close()


def create_db_tables(dbname):
    try:
        # Connect to an existing database
        conn2 = psycopg2.connect(user="postgres",
                                 password="postgres",
                                 host="127.0.0.1",
                                 port="5432",
                                 database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur2 = conn2.cursor()
        conn2.autocommit = True
        # Print PostgreSQL details
        Q_CREATE_TABLES = """CREATE TABLE globals (estimated_GSD REAL,
                                                  mean_flying_height REAL,
                                                  EPSG INTEGER,
                                                  X_shift DOUBLE PRECISION,
                                                  Y_shift DOUBLE PRECISION);
                             CREATE TABLE image (img_id SERIAL PRIMARY KEY,
                                                 img_name TEXT,
                                                 img_param_id INTEGER);
                             CREATE TABLE image_param (img_param_id SERIAL PRIMARY KEY,
                                                       nrows INTEGER,
                                                       ncols INTEGER,
                                                       X DOUBLE PRECISION,
                                                       Y DOUBLE PRECISION,
                                                       Z DOUBLE PRECISION,
                                                       camera_matrix DOUBLE PRECISION [3][3],
                                                       rotation_matrix DOUBLE PRECISION [3][3],
                                                       geobound DOUBLE PRECISION [4][3]);
                             CREATE TABLE image_pair (img_pair_id SERIAL PRIMARY KEY,
                                                      base_img_id INTEGER NOT NULL,
                                                      side_img_id INTEGER NOT NULL,
                                                      overlap REAL,
                                                      rect_param_id INTEGER,
                                                      UNIQUE (base_img_id, side_img_id));
                             CREATE TABLE rectification_param (rect_param_id SERIAL PRIMARY KEY,
                                                               H1 DOUBLE PRECISION [3][3],
                                                               H2 DOUBLE PRECISION [3][3],
                                                               H_shift DOUBLE PRECISION [3][3],
                                                               K DOUBLE PRECISION [3][3],
                                                               R DOUBLE PRECISION [3][3]);
                             CREATE TABLE multi_stereo (ms_id SERIAL PRIMARY KEY,
                                                        base_img_id INTEGER NOT NULL,
                                                        overlapping_bounds INTEGER [],
                                                        pc_ready BOOLEAN,
                                                        pc_id INTEGER);
                             CREATE TABLE point_cloud (pc_id SERIAL PRIMARY KEY,
                                                       pc_name TEXT,
                                                       footprint DOUBLE PRECISION [4][2],
                                                       n_points INTEGER);"""
        cur2.execute(Q_CREATE_TABLES)
        print("db tables created")
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn2):
            cur2.close()
            conn2.close()


def populate_imgparams_table(img_params, estimated_GSD, dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor()
        conn.autocommit = True
        Q_INSERT_IMG_PARAMS = """INSERT INTO image_param
                                   (nrows, ncols,
                                    X, Y, Z,
                                    camera_matrix, rotation_matrix,
                                    geobound)
                                 VALUES
                                   ({nrows}, {ncols},
                                    {X}, {Y}, {Z},
                                    '{camera_matrix}', '{rotation_matrix}',
                                    '{geobound}')
                                   RETURNING img_param_id;"""
        Q_INSERT_IMAGE = """INSERT INTO image
                              (img_name, img_param_id)
                              VALUES
                              ('{img_name}', {img_param_id});"""
        for img_name, params in img_params.items():
            nrows, ncols = params["nrows"], params["ncols"]
            X, Y, Z = params["X"], params["Y"], params["Z"]
            camera_matrix = str(params["camera_matrix"].tolist())
            camera_matrix = camera_matrix.replace("[", "{")
            camera_matrix = camera_matrix.replace("]", "}")
            rotation_matrix = str(params["rotation_matrix"].tolist())
            rotation_matrix = rotation_matrix.replace("[", "{")
            rotation_matrix = rotation_matrix.replace("]", "}")
            geobound = np.array(params["geobounds"])
            geobound = geobound.reshape((4, 3))
            geobound = str([i.tolist() for i in geobound])
            geobound = geobound.replace("[", "{")
            geobound = geobound.replace("]", "}")
            cur.execute(Q_INSERT_IMG_PARAMS.format(nrows=nrows, ncols=ncols,
                                                   X=X, Y=Y, Z=Z,
                                                   camera_matrix=camera_matrix,
                                                   rotation_matrix=rotation_matrix,
                                                   geobound=geobound))
            img_param_id = cur.fetchone()[0]
            cur.execute(Q_INSERT_IMAGE.format(img_name=img_name,
                                              img_param_id=img_param_id))
        print("image and image_params tables populated")
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()


def check_pair_records(left, right, dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor()
        conn.autocommit = True
        Q_GET_BASE_IMG_ID = """SELECT img_id
                               FROM image
                               WHERE img_name='{}'""".format(left["fname"])
        cur.execute(Q_GET_BASE_IMG_ID)
        base_img_id = cur.fetchone()[0]
        Q_GET_SIDE_IMG_ID = """SELECT img_id
                               FROM image
                               WHERE img_name='{}'""".format(right["fname"])
        cur.execute(Q_GET_SIDE_IMG_ID)
        side_img_id = cur.fetchone()[0]
        Q_CHECK_IMAGE_PAIR = """SELECT EXISTS
                                  (SELECT 1 FROM image_pair
                                   WHERE base_img_id={}
                                   AND side_img_id={})""".format(base_img_id,
                                                                 side_img_id)
        cur.execute(Q_CHECK_IMAGE_PAIR)
        pair_in_db = cur.fetchone()[0]
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()
    disp_fname = left["fname"][:-4] + right["fname"][:-4] + ".tif"
    disp_path = os.path.join(TMP_DISP_WARP, disp_fname)
    pair_in_disk = os.path.exists(disp_path)
    return pair_in_db, pair_in_disk


def get_img_params_db(img_name, dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        conn.autocommit = True
        Q_GET_IMG_PARAMS = """SELECT *
                              FROM image_param
                              WHERE img_param_id=
                                (SELECT img_param_id
                                 FROM image
                                 WHERE img_name='{}')""".format(img_name)
        # print(Q_GET_IMG_PARAMS)
        cur.execute(Q_GET_IMG_PARAMS)
        res = cur.fetchall()[0]
        img_params = {"nrows": res["nrows"],
                      "ncols": res["ncols"],
                      "X": res["x"],
                      "Y": res["y"],
                      "Z": res["z"],
                      "camera_matrix": np.array(res["camera_matrix"]),
                      "rotation_matrix": np.array(res["rotation_matrix"]),
                      "geobounds": np.array(res["geobound"])}
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()
    return img_params


def insert_image_pair_to_db(left, right, rect_params_L, dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor()
        conn.autocommit = True
        Q_SAVE_RECT_PARAMS = """INSERT INTO rectification_param
                                  (H1, H2, H_shift, K, R)
                                VALUES
                                  ('{H1}', '{H2}', '{H_shift}', '{K}', '{R}')
                                RETURNING rect_param_id;"""
        H1 = str(rect_params_L["H1"].tolist())
        H1 = H1.replace("[", "{")
        H1 = H1.replace("]", "}")
        H2 = str(rect_params_L["H2"].tolist())
        H2 = H2.replace("[", "{")
        H2 = H2.replace("]", "}")
        H_shift = str(rect_params_L["H_shift"].tolist())
        H_shift = H_shift.replace("[", "{")
        H_shift = H_shift.replace("]", "}")
        K = str(rect_params_L["K"].tolist())
        K = K.replace("[", "{")
        K = K.replace("]", "}")
        R = str(rect_params_L["R"].tolist())
        R = R.replace("[", "{")
        R = R.replace("]", "}")
        cur.execute(Q_SAVE_RECT_PARAMS.format(H1=H1, H2=H2, H_shift=H_shift,
                                              K=K, R=R))
        rect_param_id = cur.fetchone()[0]

        Q_GET_BASE_IMG_ID = """SELECT img_id
                               FROM image
                               WHERE img_name='{}'""".format(left["fname"])
        cur.execute(Q_GET_BASE_IMG_ID)
        base_img_id = cur.fetchone()[0]
        Q_GET_SIDE_IMG_ID = """SELECT img_id
                               FROM image
                               WHERE img_name='{}'""".format(right["fname"])
        cur.execute(Q_GET_SIDE_IMG_ID)
        side_img_id = cur.fetchone()[0]

        Q_GET_BASE_GEOBOUND = """SELECT ip.geobound
                                 FROM image_param ip
                                   LEFT JOIN image i
                                   ON ip.img_param_id=i.img_param_id
                                 WHERE i.img_name='{}'""".format(left["fname"])
        cur.execute(Q_GET_BASE_GEOBOUND)
        left_extent = np.array(cur.fetchone()[0])
        Q_GET_SIDE_GEOBOUND = """SELECT ip.geobound
                                 FROM image_param ip
                                   LEFT JOIN image i
                                   ON ip.img_param_id=i.img_param_id
                                 WHERE i.img_name='{}'""".format(right["fname"])
        cur.execute(Q_GET_SIDE_GEOBOUND)
        right_extent = np.array(cur.fetchone()[0])

        left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
        right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
        overlap = left_extent.intersection(right_extent)
        overlap_ratio = overlap.area / left_extent.area
        
        Q_SAVE_IMAGE_PAIR = """INSERT INTO image_pair
                                 (base_img_id, side_img_id,
                                  overlap, rect_param_id)
                               VALUES
                                 ({base_img_id}, {side_img_id},
                                  {overlap}, {rect_param_id});"""
        cur.execute(Q_SAVE_IMAGE_PAIR.format(base_img_id=base_img_id,
                                             side_img_id=side_img_id,
                                             overlap=overlap_ratio,
                                             rect_param_id=rect_param_id))
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()


def get_image_pair_id(left, right, dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        conn.autocommit = True
        Q_GET_IMAGE_PAIR_ID = """SELECT img_pair_id
                                 FROM image_pair ip
                                   LEFT JOIN image li
                                   ON ip.base_img_id=li.img_id
                                   LEFT JOIN image ri
                                   ON ip.side_img_id=ri.img_id
                                 WHERE li.img_name='{left_fname}'
                                   AND ri.img_name='{right_fname}'"""
        cur.execute(Q_GET_IMAGE_PAIR_ID.format(left_fname=left["fname"],
                                               right_fname=right["fname"]))
        img_pair_id = cur.fetchone()["img_pair_id"]
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()
    return img_pair_id


def get_rect_params(left, right, dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        conn.autocommit = True
        img_pair_id = get_image_pair_id(left, right, dbname)
        Q_GET_RECT_PARAMS = """SELECT * FROM
                                 rectification_param rp
                                 LEFT JOIN
                                 image_pair ip
                                 ON rp.rect_param_id=ip.rect_param_id
                               WHERE ip.img_pair_id={img_pair_id}"""
        cur.execute(Q_GET_RECT_PARAMS.format(img_pair_id=img_pair_id))
        res = cur.fetchone()
        rect_params = {"H1": np.array(res["h1"]),
                       "H2": np.array(res["h2"]),
                       "H_shift": np.array(res["h_shift"]),
                       "K": np.array(res["k"]),
                       "R": np.array(res["r"])}
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()
    return rect_params


def get_max_object_id(dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        conn.autocommit = True
        Q_SELECT_MAX_OBJ_ID = """SELECT MAX(object_point_id) AS mopid
                                 FROM pixel"""
        cur.execute(Q_SELECT_MAX_OBJ_ID)
        max_op_id = cur.fetchone()["mopid"]
        if not max_op_id:
            max_op_id = 0
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()
    return max_op_id


def check_db_same_pixel(left,
                        base_row_coord,
                        base_column_coord,
                        dbname):
    try:
        # Connect to an existing database
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))

        # Create a cursor to perform database operations
        cur = conn.cursor()
        conn.autocommit = True
        Q_GET_SAME_PIXEL = """SELECT object_point_id
                              FROM pixel p
                                LEFT JOIN image_pair ip
                                ON p.img_pair_id=ip.img_pair_id
                                LEFT JOIN image i
                                ON ip.side_img_id=i.img_id
                              WHERE i.img_name='{left_fname}'
                                AND (p.side_row_coord - {base_row_coord}) < 1
                                AND (p.side_column_coord - {base_column_coord}) < 1;"""
        cur.execute(Q_GET_SAME_PIXEL.format(left_fname=left["fname"],
                                            base_row_coord=base_row_coord,
                                            base_column_coord=base_column_coord))
        res = cur.fetchall()
        if len(res) == 0:
            correspondence = 0
        else:
            opids = list(set([i[0] for i in res]))
            if len(opids) == 1:
                correspondence = opids[0]
            else:
                raise Exception("Corresponding pixels with different opids detected!!!")
    except (Exception, Error) as error:
        print("Error while connecting to db", error)
    finally:
        if (conn):
            cur.close()
            conn.close()
    return correspondence


def insert_pixels_to_db(left, right, dbname,
                        orig_disp_array1, orig_mask_array1, mappings):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))

    # Create a cursor to perform database operations
    cur = conn.cursor()
    img_pair_id = get_image_pair_id(left, right, dbname)
    # max_object_id = get_max_object_id(dbname)
    left_img_params = get_img_params_db(left["fname"], dbname)
    right_img_params = get_img_params_db(right["fname"], dbname)
    R1 = left_img_params["rotation_matrix"]
    R2 = right_img_params["rotation_matrix"]
    Z1 = np.array([left_img_params["X"], left_img_params["Y"], left_img_params["Z"]])
    Z2 = np.array([right_img_params["X"], right_img_params["Y"], right_img_params["Z"]])
    K_left = left_img_params["camera_matrix"]
    x_c_orig, y_c_orig = K_left[0, 2], K_left[1][2]
    left_mapping = mappings["left"]
    right_mapping = mappings["right"]
    # extract overlap mask
    left_extent = left_img_params["geobounds"]
    right_extent = right_img_params["geobounds"]
    mean_flying_height = left_img_params["mean_flying_height"]
    mean_z = Z1[-1] - mean_flying_height
    left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
    right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
    overlap = left_extent.intersection(right_extent)
    xx, yy = overlap.exterior.coords.xy
    xx = xx.tolist()
    yy = yy.tolist()
    overlap_coords_world = [(x, y, mean_z, 1) for x, y in zip(xx, yy)]
    overlap_coords_world = overlap_coords_world[:-1]
    t = -R1@Z1
    R_t = np.zeros((3, 4), dtype=np.float64)
    R_t[:, :-1] = R1
    R_t[:, -1] = t
    P = K_left@R_t
    overlap_coords_image = [P@np.array(i) for i in overlap_coords_world]
    overlap_coords_image = [i / i[-1] for i in overlap_coords_image]
    overlap_bounds = Polygon([i[:-1] for i in overlap_coords_image])
    overlap_bounds = overlap_bounds.buffer(-OVERLAP_BOUNDS_BUFFER)
    mask = rasterio.features.rasterize([overlap_bounds], out_shape=orig_disp_array1.shape)
    # add overlap mask
    orig_mask_array1 = orig_mask_array1 * mask
    # check if pixels were written already for image pair
    Q_COUNT_PIXELS_IMG_PAIR = """SELECT COUNT(*)
                                 FROM pixel
                                 WHERE img_pair_id={}""".format(img_pair_id)
    cur.execute(Q_COUNT_PIXELS_IMG_PAIR)
    pixel_count = cur.fetchone()[0]
    if pixel_count == (orig_mask_array1==1).sum():
        print("pixels for image pair {}, {} already written in db".format(left["fname"],
                                                                          right["fname"]))
    else:
        if pixel_count > 0:
            print("pixel count for image pair {}, {} inconsistent, \
deleting previous db records...".format(left["fname"], right["fname"]))
            Q_DELETE_PIXELS_IMG_PAIR = """DELETE FROM pixel
                                          WHERE img_pair_id={}""".format(img_pair_id)
            cur.execute(Q_DELETE_PIXELS_IMG_PAIR)
            conn.commit()
        # calculate base and side image coordinates
        ny, nx = orig_disp_array1.shape
        x = np.linspace(0, nx - 1, nx)
        y = np.linspace(0, ny - 1, ny)
        xv, yv = np.meshgrid(x, y)
        xv = xv[orig_mask_array1==1]
        yv = yv[orig_mask_array1==1]
        disp_values = orig_disp_array1[orig_mask_array1==1]
        third_column = np.ones(xv.shape)
        left_data = np.stack((xv, yv, third_column, disp_values))
        right_data = left_data.copy()
        # map to warped space of left image
        right_data[0:3, :] = np.linalg.inv(left_mapping) @ right_data[0:3, :]
        right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
        # add x-disparity
        right_data[0:1, :] = right_data[0:1, :] - right_data[-1:, :]
        # map to orig space of right image
        right_data[0:3, :] = right_mapping @ right_data[0:3, :]
        right_data[0:3, :] = right_data[0:3, :] / right_data[2:3, :]
        left_coords = left_data[0:2]
        right_coords = right_data[0:2]
        del left_data, right_data
        # shift coordinate based on principal point coordinate
        left_coords[0] = left_coords[0] - x_c_orig
        left_coords[1] = left_coords[1] - y_c_orig
        right_coords[0] = right_coords[0] - x_c_orig
        right_coords[1] = right_coords[1] - y_c_orig
        masked_image = left["array"][orig_mask_array1==1]
        masked_image = masked_image.reshape(-1, 3)
        pixels = [str((masked_image[i, 0],
                       masked_image[i, 1],
                       masked_image[i, 2],
                       img_pair_id,
                       left_coords[1, i],
                       left_coords[0, i],
                       disp_values[i],
                       right_coords[1, i],
                       right_coords[0, i])) for i in range(disp_values.size)]
        sio = StringIO()
        def format_entry(row):
            row = row.replace("(", "")
            row = row.replace(")", "")
            row = row.replace(",", "\t")
            return row
        sio.write('\n'.join(format_entry(pixel) for pixel in pixels))
        sio.seek(0)
        cur.copy_from(sio, "pixel", columns=("red", "green", "blue",
                                             "img_pair_id",
                                             "base_row_coord",
                                             "base_column_coord",
                                             "disparity",
                                             "side_row_coord",
                                             "side_column_coord"))
        conn.commit()
        print("{} pixels written in db".format(len(pixels)))
    cur.close()
    conn.close()


def get_save_disp_rect_params(left, right, dbname,
                              DF, disparity_method):
    rect_params_L = None
    disp_array_L = None
    pair_in_db, pair_in_disk = check_pair_records(left, right, dbname)
    if pair_in_disk and pair_in_db:
        print("image pair {}, {} found both in disk and db".format(left["fname"],
                                                                   right["fname"]))
        disp_fname_L = left["fname"][:-4] + right["fname"][:-4] + ".tif"
        disp_path_L = os.path.join(TMP_DISP_WARP, disp_fname_L)
        disp_img_L = gdal.Open(disp_path_L)
        disp_array_L = disp_img_L.ReadAsArray()
    elif pair_in_db and not pair_in_disk:
        print("image pair {}, {} found in db but not in disk".format(left["fname"],
                                                                     right["fname"]))
        disp_arrays_L, rect_params_L = get_left_disparity_db(left, right,
                                                             DF, disparity_method,
                                                             dbname)
        disp_array_L = disp_arrays_L[:, :, 0]
        disp_fname_L = left["fname"][:-4] + right["fname"][:-4] + ".tif"
        disp_path_L = os.path.join(TMP_DISP_WARP, disp_fname_L)
        write_array_float(np.expand_dims(disp_array_L, axis=-1), disp_path_L)
    elif pair_in_disk and not pair_in_db:
        print("image pair {}, {} found in disk but not in db".format(left["fname"],
                                                                     right["fname"]))
        left_img_params = get_img_params_db(left["fname"], dbname)
        right_img_params = get_img_params_db(right["fname"], dbname)
        # rectify image pairs
        print("rectifying {} and {} image pair...".format(left["fname"], right["fname"]))
        img_rect_params_dict = rectify_stereopair(left["array"], right["array"],
                                                  left_img_params, right_img_params,
                                                  DF)
        rect_img1, rect_img2 = img_rect_params_dict["image_pairs"]
        rect_params = img_rect_params_dict["rect_params"]
        insert_image_pair_to_db(left, right, rect_params, dbname)
    else:
        print("image pair {}, {} not found in disk and db".format(left["fname"],
                                                                  right["fname"]))
        disp_arrays_L, rect_params_L = get_left_disparity_db(left, right,
                                                             DF, disparity_method,
                                                             dbname)
        insert_image_pair_to_db(left, right, rect_params_L, dbname)
        disp_array_L = disp_arrays_L[:, :, 0]
        disp_fname_L = left["fname"][:-4] + right["fname"][:-4] + ".tif"
        disp_path_L = os.path.join(TMP_DISP_WARP, disp_fname_L)
        write_array_float(np.expand_dims(disp_array_L, axis=-1), disp_path_L)
    if disp_array_L is None:
        disp_fname_L = left["fname"][:-4] + right["fname"][:-4] + ".tif"
        disp_path_L = os.path.join(TMP_DISP_WARP, disp_fname_L)
        disp_img_L = gdal.Open(disp_path_L)
        disp_array_L = disp_img_L.ReadAsArray()
    if rect_params_L is None:
        rect_params_L = get_rect_params(left, right, dbname)
    return disp_array_L, rect_params_L


def insert_multi_stereo(dbname, base_image, side_images):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    imgparams1 = get_img_params_db(base_image, dbname)
    # R1 = imgparams1["rotation_matrix"]
    # Z1 = np.array([imgparams1["X"], imgparams1["Y"], imgparams1["Z"]])
    base_extent = imgparams1["geobounds"]
    base_extent = Polygon([tuple(i[:-1]) for i in base_extent])
    K = imgparams1["camera_matrix"]
    c = K[0][0]
    if len(side_images) > 1:
        side_extents = []
        for idx, side_image in enumerate(side_images):
            imgparams2 = get_img_params_db(side_image, dbname)
            # R2 = imgparams2["rotation_matrix"]
            # Z2 = np.array([imgparams2["X"], imgparams2["Y"], imgparams2["Z"]])
            # calculate image coordinates of overlap and add mask based on it
            side_extent = imgparams2["geobounds"]
            side_extent = Polygon([tuple(i[:-1]) for i in side_extent])
            side_extents.append(side_extent)
        side_image_overlap = None
        for j, side_extent  in enumerate(side_extents[:-1]):
            if j == 0:
                side_extent_A = side_extent
                side_extent_B = side_extents[j+1]
            else:
                side_extent_A = side_image_overlap
                side_extent_B = side_extents[j+1]
            side_image_overlap = side_extent_A.intersection(side_extent_B)
    else:
        side_image = side_images[0]
        imgparams2 = get_img_params_db(side_image, dbname)
        side_extent = imgparams2["geobounds"]
        side_extent = Polygon([tuple(i[:-1]) for i in side_extent])
        side_image_overlap = side_extent.intersection(side_extent)
    final_bounds = base_extent.intersection(side_image_overlap)
    Q_GET_GSD = """SELECT estimated_gsd FROM globals;"""
    cur.execute(Q_GET_GSD)
    gsd = float(cur.fetchone()[0])
    final_bounds = final_bounds.buffer(-OVERLAP_BOUNDS_BUFFER*gsd)
    Q_GET_ALL_IMG_NAME_ID = """SELECT img_id, img_name FROM image;"""
    cur.execute(Q_GET_ALL_IMG_NAME_ID)
    overlapping_ids = []
    for img_id, img_name in cur.fetchall():
        img_extent = get_img_params_db(img_name, dbname)["geobounds"]
        img_extent = Polygon([tuple(i[:-1]) for i in img_extent])
        if final_bounds.intersects(img_extent):
            overlapping_ids.append(img_id)
    Q_INSERT_MULTI_STEREO = """INSERT INTO multi_stereo
                                 (base_img_id,
                                  overlapping_bounds)
                               VALUES
                                 ({base_img_id},
                                  ARRAY {overlapping_bounds});"""
    Q_GET_BASE_IMG_ID = """SELECT img_id
                           FROM image
                           WHERE img_name='{}'""".format(base_image)
    cur.execute(Q_GET_BASE_IMG_ID)
    base_img_id = cur.fetchone()[0]
    print("inserting multi-stereo record to db...")
    cur.execute(Q_INSERT_MULTI_STEREO.format(base_img_id=base_img_id,
                                             overlapping_bounds=overlapping_ids))
    cur.close()
    conn.close()


def check_multi_stereo_entry(dbname, base_image):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    Q_CHECK_MULTI_STEREO_ENTRY = """SELECT
                                      EXISTS(
                                        SELECT 1
                                        FROM multi_stereo ms
                                        LEFT JOIN image im
                                          ON ms.base_img_id=im.img_id
                                        WHERE im.img_name='{}');"""
    cur.execute(Q_CHECK_MULTI_STEREO_ENTRY.format(base_image))
    mse_exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    return mse_exists


def get_global_param(dbname, param_name):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    Q_GET_GLOBAL_PARAM = """SELECT {} FROM globals LIMIT 1;"""
    cur.execute(Q_GET_GLOBAL_PARAM.format(param_name))
    param = cur.fetchone()[0]
    cur.close()
    conn.close()
    return param


def check_point_cloud_entry(dbname, base_image):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    Q_CHECK_POINT_CLOUD_ENTRY = """SELECT
                                     EXISTS(
                                       SELECT 1
                                       FROM point_cloud pc
                                       LEFT JOIN multi_stereo ms
                                         ON ms.pc_id=pc.pc_id
                                       LEFT JOIN image im
                                         ON ms.base_img_id=im.img_id
                                       WHERE im.img_name='{}');"""
    cur.execute(Q_CHECK_POINT_CLOUD_ENTRY.format(base_image))
    pc_exists = cur.fetchone()[0]
    cur.close()
    conn.close()
    return pc_exists


def insert_point_cloud_entry(dbname, base_image, points):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    Q_INSERT_POINT_CLOUD_ENTRY = """INSERT INTO point_cloud
                                      (pc_name, footprint, n_points)
                                    VALUES
                                      ('{pc_name}', '{footprint}', {npoints})
                                    RETURNING pc_id;"""
    pc_name = "{}.ply".format(base_image[:-4])
    x_min = points[:, 0].min()
    x_max = points[:, 0].max()
    y_min = points[:, 1].min()
    y_max = points[:, 1].max()
    footprint = [[x_min, y_min],
                 [x_min, y_max],
                 [x_max, y_max],
                 [x_max, y_min]]
    # write_point_cloud_bounds(base_image, footprint)
    footprint = str(footprint)
    footprint = footprint.replace("[", "{")
    footprint = footprint.replace("]", "}")
    npoints = points.shape[0]
    cur.execute(Q_INSERT_POINT_CLOUD_ENTRY.format(pc_name=pc_name,
                                                  footprint=footprint,
                                                  npoints=npoints))
    pc_id = cur.fetchone()[0]
    cur.close()
    conn.close()
    return pc_id


def update_pc_status_in_ms(dbname, base_image, pc_id):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    Q_UPDATE_PC_STATUS_IN_MS = """UPDATE multi_stereo
                                  SET pc_id={pc_id},
                                      pc_ready=True
                                  WHERE base_img_id=(SELECT img_id
                                                     FROM image
                                                     WHERE img_name='{base_image}');"""
    cur.execute(Q_UPDATE_PC_STATUS_IN_MS.format(pc_id=pc_id,
                                                base_image=base_image))
    cur.close()
    conn.close()


def get_image_filenames(dbname):
    # Connect to an existing database
    conn = psycopg2.connect(user="postgres",
                            password="postgres",
                            host="127.0.0.1",
                            port="5432",
                            database="{}".format(dbname))
    conn.autocommit = True
    # Create a cursor to perform database operations
    cur = conn.cursor()
    Q_SELECT_IMAGE_FNAMES = """SELECT img_name FROM image;"""
    cur.execute(Q_SELECT_IMAGE_FNAMES)
    imagefilenames = [i[0] for i in cur.fetchall()]
    cur.close()
    conn.close()
    return imagefilenames


def write_point_cloud_bounds(base_image, footprint):
    footprint_formatted = []
    for corner in footprint:
        footprint_formatted.append(corner)
    footprint_formatted.append(footprint[0])
    json_bounds = {"type": "FeatureCollection",
                   "features": [{"type": "Feature",
                                 "geometry": {"type": "Polygon",
                                              "coordinates": footprint_formatted}, "properties": {}}]}
    json_object = json.dumps(json_bounds, indent=4)
    footprint_path = os.path.join(TMP_PLY_BOUNDS, "{}.json".format(base_image[:-4]))
    with open(footprint_path, "w") as f:
        f.write(json_object)


def stereo_reconstruct(LEFT, RIGHT, img_params, DF, disparity_method,
                       mask_method, disparity_difference_cutoff, pair_idx):
    # # perform stereo reconstruction of LEFT and RIGHT image
    # LEFT_IMG = LEFT["fname"]
    # LEFT_IMG_OBJECT = LEFT["object"]
    # LEFT_IMG_ARRAY = LEFT["array"]
    # RIGHT_IMG = RIGHT["fname"]
    # RIGHT_IMG_OBJECT = RIGHT["object"]
    # RIGHT_IMG_ARRAY = RIGHT["array"]
    # LEFT_IMG_PARAMS = img_params[LEFT_IMG]
    # RIGHT_IMG_PARAMS = img_params[RIGHT_IMG]

    # print("rectifying {} and {} image pair...".format(LEFT_IMG, RIGHT_IMG))
    # img_rect_params_dict = rectify_stereopair(LEFT_IMG_ARRAY, RIGHT_IMG_ARRAY,
    #                                           LEFT_IMG_PARAMS, RIGHT_IMG_PARAMS,
    #                                           DF)
    # rect_img1, rect_img2 = img_rect_params_dict["image_pairs"]
    # rect_params = img_rect_params_dict["rect_params"]
    # orig_img1, orig_img2 = LEFT_IMG_OBJECT, RIGHT_IMG_OBJECT
    # orig_imgarray1 = LEFT_IMG_ARRAY
    # t = time.time()
    # t_mins = (t - T0) / 60.0
    # print("done in {:.2f} mins".format(t_mins))

    # print("reducing absolute disparity values...")
    # disparity_shift = calc_disp_reduction(LEFT_IMG_PARAMS, RIGHT_IMG_PARAMS, rect_params)
    # shifted_rect_img2 = shift_right_image(rect_img2, -disparity_shift)
    # # fig, ax = plt.subplots(nrows=1, ncols=2)
    # # ax[0].imshow(rect_img2)
    # # ax[1].imshow(shifted_rect_img2)
    # # plt.show()
    # t = time.time()
    # t_mins = (t - T0) / 60.0
    # print("done in {:.2f} mins".format(t_mins))

    # print("subsetting image pairs...")
    # img1_subsets = subset_image(rect_img1, SUBSET_HEIGHT, SUBSET_WIDTH,
    #                             SUBSET_HEIGHT_OVERLAP, SUBSET_WIDTH_OVERLAP)
    # img2_subsets = subset_image(shifted_rect_img2, SUBSET_HEIGHT, SUBSET_WIDTH,
    #                             SUBSET_HEIGHT_OVERLAP, SUBSET_WIDTH_OVERLAP)
    # t = time.time()
    # t_mins = (t - T0) / 60.0
    # print("done in {:.2f} mins".format(t_mins))


    # print("calculating disparity image...")
    # disp_arrays = stitch_disparity(rect_img1, img1_subsets, img2_subsets, disparity_method)
    if mask_method is None:
        disp_arrays, rect_params = get_left_disparity(LEFT, RIGHT, img_params, DF, disparity_method)
        disp_array1 = disp_arrays[:, :, 0]
        disp_fname = LEFT["fname"][:-4] + RIGHT["fname"][:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_WARP, disp_fname)
        write_array_float(np.expand_dims(disp_array1, axis=-1), disp_path)
        mask_array1 = np.ones((disp_array1.shape[0], disp_array1.shape[1]))
    elif mask_method == "FM":
        disp_arrays, rect_params = get_left_disparity(LEFT, RIGHT, img_params, DF, disparity_method)
        disp_array1 = disp_arrays[:, :, 0]
        disp_fname = LEFT["fname"][:-4] + RIGHT["fname"][:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_WARP, disp_fname)
        write_array_float(np.expand_dims(disp_array1, axis=-1), disp_path)
        mask_array1 = disp_arrays[:, :, 1]
    elif mask_method == "RLCC":
        disp_arrays_L, rect_params_L = get_left_disparity(LEFT, RIGHT, img_params, DF, disparity_method)
        disp_array_L = disp_arrays_L[:, :, 0]
        disp_fname_L = LEFT["fname"][:-4] + RIGHT["fname"][:-4] + ".tif"
        disp_path_L = os.path.join(TMP_DISP_WARP, disp_fname_L)
        write_array_float(np.expand_dims(disp_array_L, axis=-1), disp_path_L)
        disp_arrays_R, rect_params_R = get_left_disparity(RIGHT, LEFT, img_params, DF, disparity_method)
        disp_array_R = disp_arrays_R[:, :, 0]
        orig_disp_array_R, mappings = warp_left_disparity(disp_array_R, RIGHT["array"].shape[:-1], rect_params_R)
        orig_disp_array_R = np.expand_dims(orig_disp_array_R, axis=-1)
        disp_fname = RIGHT_IMG[:-4] + LEFT_IMG[:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
        write_array_float(orig_disp_array_R, disp_path)
        img_rect_params_dict_swap = rectify_stereopair(LEFT["array"], orig_disp_array_R,
                                                       img_params[LEFT["fname"]],
                                                       img_params[RIGHT["fname"]],
                                                       DF)
        ____, disp_array_WR = img_rect_params_dict_swap["image_pairs"]
        disp_fname_WR = RIGHT["fname"][:-4] + LEFT["fname"][:-4] + "_warped_RL.tif"
        disp_path = os.path.join(TMP_DISP_WARP, disp_fname_WR)
        write_array_float(disp_array_WR, disp_path)
        deviations = compare_disparity_maps(disp_array_L, disp_array_WR)
        deviations_fname = LEFT["fname"][:-4] + RIGHT["fname"][:-4] + "_deviations.tif"
        deviations_path = os.path.join(TMP_DISP_WARP, deviations_fname)
        write_array_float(np.expand_dims(deviations, axis=-1), deviations_path)
        # mask disparities greater than difference threshold
        reclass_deviations = np.zeros(deviations.shape)
        reclass_deviations[np.absolute(deviations) < disparity_difference_cutoff] = 1
        rect_params = rect_params_L
        disp_array1 = disp_array_L
        mask_array1 = reclass_deviations
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    print("warping disparity image back to original space...")
    orig_disp_array1, mappings = warp_left_disparity(disp_array1, LEFT["array"].shape[:-1], rect_params)
    disp_fname = LEFT["fname"][:-4] + RIGHT_IMG[:-4] + ".tif"
    disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
    write_array_float(np.expand_dims(orig_disp_array1, axis=-1), disp_path)
    orig_mask_array1, mappings = warp_left_disparity(mask_array1, LEFT["array"].shape[:-1], rect_params)
    mask_fname = LEFT["fname"][:-4] + RIGHT_IMG[:-4] + "_mask.tif"
    mask_path = os.path.join(TMP_DISP_ORIG, mask_fname)
    write_array_float(np.expand_dims(orig_mask_array1, axis=-1), mask_path)
    # # # use disparity estimator mask
    # # mask_array1 =  disp_arrays[:, :, 1]
    # # hack to specify mask
    # # use R-L consistency
    # mask_fname = "reclass_deviations_t1.tif"
    # mask_path = os.path.join(TMP_DISP_WARP, mask_fname)
    # mask_img = gdal.Open(mask_path)
    # mask_array1 = mask_img.ReadAsArray()
    # # plt.imshow(mask_array1)
    # # plt.show()
    # disp_array1 = disp_array1 + disparity_shift
    # disp_fname = LEFT_IMG[:-4] + RIGHT_IMG[:-4] + ".tif"
    # disp_path = os.path.join(TMP_DISP_WARP, disp_fname)
    # write_array_float(np.expand_dims(disp_array1, axis=-1), disp_path)
    # # plt.imshow(disp_array1)
    # # plt.show()
    # t = time.time()
    # t_mins = (t - T0) / 60.0
    # print("done in {:.2f} mins".format(t_mins))

    # print("warping disparity image back to original space...")
    # orig_disp_array1, mappings = warp_left_disparity(disp_array1, LEFT.shape[:-1], rect_params)
    # disp_fname = LEFT_IMG[:-4] + RIGHT_IMG[:-4] + ".tif"
    # disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
    # write_array_float(np.expand_dims(orig_disp_array1, axis=-1), disp_path)
    # right_disp_array = np.expand_dims(orig_disp_array1, axis=-1)
    # img_rect_params_dict_swap = rectify_stereopair(RIGHT_IMG_ARRAY, right_disp_array,
    #                                                RIGHT_IMG_PARAMS, LEFT_IMG_PARAMS,
    #                                                DF)
    # ____, rect_disp1_swap = img_rect_params_dict_swap["image_pairs"]
    # disp_fname = LEFT_IMG[:-4] + RIGHT_IMG[:-4] + "_warped_RL.tif"
    # disp_path = os.path.join(TMP_DISP_WARP, disp_fname)
    # write_array_float(rect_disp1_swap, disp_path)
    # plt.imshow(orig_disp_array1)
    # plt.show()
    # add warped mask
    # orig_mask_array1, mappings = warp_left_disparity(mask_array1, orig_imgarray1.shape[:-1], rect_params)
    # mask_fname = LEFT_IMG[:-4] + RIGHT_IMG[:-4] + "_mask.tif"
    # mask_path = os.path.join(TMP_DISP_ORIG, mask_fname)
    # write_array_float(np.expand_dims(orig_mask_array1, axis=-1), mask_path)

    # # hack don't use warped mask
    # orig_mask_array1[:, :] = 1

    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    print("triangulating using left disparity image...")
    points_3D, mask = triangulate_left(LEFT["fname"], RIGHT["fname"], orig_disp_array1, orig_mask_array1,
                                       mappings, rect_params, img_params)
    # points_3D, mask = triangulate_left2(LEFT["fname"], RIGHT["fname"], orig_disp_array1, orig_mask_array1,
    #                                     mappings, rect_params, img_params)
    # points_3D, mask = left_disparity_to_pointcloud(LEFT["fname"], RIGHT["fname"],
    #                                                disp_array1, orig_disp_array1,
    #                                                rect_params, img_params)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    print("writing ply file...")
    # mask using inner-buffered overlap bounds
    # add mask based on occlusion/validity/RLCC
    total_mask = mask * orig_mask_array1
    # # don't add mask
    # total_mask = mask

    masked_image = LEFT["array"][total_mask==1]
    
    # masked_image = rect_img1[mask == 1]
    points_3D[:, 3:] = (255 * masked_image.reshape(-1, 3)).astype(np.uint8)
    write_ply(points_3D[:, 0:3], points_3D[:, 3:], TMP_LEFT_PLY, pair_idx)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))
    # return points_3D[:, 0:3]
    return TMP_LEFT_PLY


def mvs_reconstruct(multiview_dict, img_params, DF, disparity_method,
                    disparity_difference_cutoff, pair_idx):
    LEFT_IMG = list(multiview_dict.keys())[0]
    base_image = LEFT_IMG
    side_images = multiview_dict[LEFT_IMG]
    # print(LEFT_IMG)
    # print(side_images)
    disp_arrays = []
    mask_arrays = []
    rect_params_list = []
    mappings_list = []
    for RIGHT_IMG in side_images:
        imgpath = os.path.join(IMGDIR, LEFT_IMG)
        img = gdal.Open(imgpath)
        LEFT_IMG_OBJECT = img
        LEFT_IMG_ARRAY = np.transpose(img.ReadAsArray(), [1, 2, 0])
        imgpath = os.path.join(IMGDIR, RIGHT_IMG)
        img = gdal.Open(imgpath)
        RIGHT_IMG_OBJECT = img
        RIGHT_IMG_ARRAY = np.transpose(img.ReadAsArray(), [1, 2, 0])
        LEFT = {"fname": LEFT_IMG,
                "object": LEFT_IMG_OBJECT,
                "array": LEFT_IMG_ARRAY}
        RIGHT = {"fname": RIGHT_IMG,
                 "object": RIGHT_IMG_OBJECT,
                 "array": RIGHT_IMG_ARRAY}
        disp_arrays_L, rect_params_L = get_left_disparity(LEFT, RIGHT, img_params, DF, disparity_method)
        disp_array_L = disp_arrays_L[:, :, 0]
        disp_fname_L = LEFT["fname"][:-4] + RIGHT["fname"][:-4] + ".tif"
        disp_path_L = os.path.join(TMP_DISP_WARP, disp_fname_L)
        write_array_float(np.expand_dims(disp_array_L, axis=-1), disp_path_L)
        disp_arrays_R, rect_params_R = get_left_disparity(RIGHT, LEFT, img_params, DF, disparity_method)
        disp_array_R = disp_arrays_R[:, :, 0]
        orig_disp_array_R, mappings = warp_left_disparity(disp_array_R, RIGHT["array"].shape[:-1], rect_params_R)
        orig_disp_array_R = np.expand_dims(orig_disp_array_R, axis=-1)
        disp_fname = RIGHT_IMG[:-4] + LEFT_IMG[:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
        write_array_float(orig_disp_array_R, disp_path)
        img_rect_params_dict_swap = rectify_stereopair(LEFT["array"], orig_disp_array_R,
                                                       img_params[LEFT["fname"]],
                                                       img_params[RIGHT["fname"]],
                                                       DF)
        ____, disp_array_WR = img_rect_params_dict_swap["image_pairs"]
        disp_fname_WR = RIGHT["fname"][:-4] + LEFT["fname"][:-4] + "_warped_RL.tif"
        disp_path = os.path.join(TMP_DISP_WARP, disp_fname_WR)
        write_array_float(disp_array_WR, disp_path)
        deviations = compare_disparity_maps(disp_array_L, disp_array_WR)
        deviations_fname = LEFT["fname"][:-4] + RIGHT["fname"][:-4] + "_deviations.tif"
        deviations_path = os.path.join(TMP_DISP_WARP, deviations_fname)
        write_array_float(np.expand_dims(deviations, axis=-1), deviations_path)
        # mask disparities greater than difference threshold
        reclass_deviations = np.zeros(deviations.shape)
        reclass_deviations[np.absolute(deviations) < disparity_difference_cutoff] = 1
        rect_params = rect_params_L
        disp_array1 = disp_array_L
        mask_array1 = reclass_deviations
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))

        print("warping disparity image back to original space...")
        orig_disp_array1, mappings = warp_left_disparity(disp_array1, LEFT["array"].shape[:-1], rect_params)
        disp_fname = LEFT["fname"][:-4] + RIGHT_IMG[:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
        write_array_float(np.expand_dims(orig_disp_array1, axis=-1), disp_path)
        orig_mask_array1, mappings = warp_left_disparity(mask_array1, LEFT["array"].shape[:-1], rect_params)
        mask_fname = LEFT["fname"][:-4] + RIGHT_IMG[:-4] + "_mask.tif"
        mask_path = os.path.join(TMP_DISP_ORIG, mask_fname)
        write_array_float(np.expand_dims(orig_mask_array1, axis=-1), mask_path)
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))

        disp_arrays.append(orig_disp_array1)
        mask_arrays.append(orig_mask_array1)
        rect_params_list.append(rect_params_L)
        mappings_list.append(mappings)

    print("triangulating using left disparity image...")
    points_3D, mask = triangulate_multi(base_image, side_images, disp_arrays, mask_arrays,
                                        mappings_list, rect_params_list, img_params)

    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    print("writing ply file...")
    # mask using inner-buffered overlap bounds
    # add mask based on occlusion/validity/RLCC
    total_mask = mask * orig_mask_array1

    masked_image = LEFT["array"][total_mask==1]
    
    points_3D[:, 3:] = (masked_image.reshape(-1, 3)).astype(np.uint8)
    write_ply(points_3D[:, 0:3], points_3D[:, 3:], TMP_LEFT_PLY, pair_idx)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))
    # return points_3D[:, 0:3]
    return TMP_LEFT_PLY


def mvs_reconstruct_db(multiview_dict, DF, disparity_method,
                       disparity_difference_cutoff, dbname):
    left_img = list(multiview_dict.keys())[0]
    base_image = left_img
    side_images = multiview_dict[left_img]
    disp_arrays = []
    mask_arrays = []
    rect_params_list = []
    mappings_list = []
    for right_img in side_images:
        imgpath = os.path.join(IMGDIR, left_img)
        img = gdal.Open(imgpath)
        left_img_array = np.transpose(img.ReadAsArray(), [1, 2, 0])
        imgpath = os.path.join(IMGDIR, right_img)
        img = gdal.Open(imgpath)
        right_img_array = np.transpose(img.ReadAsArray(), [1, 2, 0])
        left = {"fname": left_img,
                "array": left_img_array}
        right = {"fname": right_img,
                 "array": right_img_array}
        disp_array_L, rect_params_L = get_save_disp_rect_params(left, right, dbname,
                                                                DF, disparity_method)
        disp_array_R, rect_params_R = get_save_disp_rect_params(right, left, dbname,
                                                                DF, disparity_method)

        print("creating mask based on right-left consistency check...")
        disp_fname = right["fname"][:-4] + left["fname"][:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
        if not os.path.exists(disp_path):
            orig_disp_array_R, mappings = warp_left_disparity(disp_array_R,
                                                              right["array"].shape[:-1],
                                                              rect_params_R)
            orig_disp_array_R = np.expand_dims(orig_disp_array_R, axis=-1)
            write_array_float(orig_disp_array_R, disp_path)
        else:
            disp_img = gdal.Open(disp_path)
            orig_disp_array_R = disp_img.ReadAsArray()
            orig_disp_array_R = np.expand_dims(orig_disp_array_R, axis=-1)
        disp_fname_WR = right["fname"][:-4] + left["fname"][:-4] + "_warped_RL.tif"
        disp_path = os.path.join(TMP_DISP_WARP, disp_fname_WR)
        if not os.path.exists(disp_path):
            img_rect_params_dict_swap = rectify_stereopair(left["array"], orig_disp_array_R,
                                                           get_img_params_db(left["fname"],
                                                                             dbname),
                                                           get_img_params_db(right["fname"],
                                                                             dbname),
                                                           DF)
            ____, disp_array_WR = img_rect_params_dict_swap["image_pairs"]
            write_array_float(disp_array_WR, disp_path)
        else:
            disp_img = gdal.Open(disp_path)
            disp_array_WR = disp_img.ReadAsArray()
        deviations_fname = left["fname"][:-4] + right["fname"][:-4] + "_deviations.tif"
        deviations_path = os.path.join(TMP_DISP_WARP, deviations_fname)
        if not os.path.exists(deviations_path):
            deviations = compare_disparity_maps(disp_array_L, disp_array_WR)
            write_array_float(np.expand_dims(deviations, axis=-1), deviations_path)
        else:
            deviations_img = gdal.Open(deviations_path)
            deviations = deviations_img.ReadAsArray()
        # mask disparities greater than difference threshold
        reclass_deviations = np.zeros(deviations.shape)
        reclass_deviations[np.absolute(deviations) < disparity_difference_cutoff] = 1
        rect_params = rect_params_L
        disp_array1 = disp_array_L
        mask_array1 = reclass_deviations
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))

        print("warping disparity image back to original space...")
        disp_fname = left["fname"][:-4] + right["fname"][:-4] + ".tif"
        disp_path = os.path.join(TMP_DISP_ORIG, disp_fname)
        if not os.path.exists(disp_path):
            orig_disp_array1, mappings = warp_left_disparity(disp_array1,
                                                             left["array"].shape[:-1],
                                                             rect_params)
            write_array_float(np.expand_dims(orig_disp_array1, axis=-1), disp_path)
        else:
            disp_img = gdal.Open(disp_path)
            orig_disp_array1 = disp_img.ReadAsArray()
        mask_fname = left["fname"][:-4] + right["fname"][:-4] + "_mask.tif"
        mask_path = os.path.join(TMP_DISP_ORIG, mask_fname)
        orig_mask_array1, mappings = warp_left_disparity(mask_array1,
                                                         left["array"].shape[:-1],
                                                         rect_params)
        if not os.path.exists(mask_path):
            write_array_float(np.expand_dims(orig_mask_array1, axis=-1), mask_path)
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))
        # print("writing pixels to db")
        # insert_pixels_to_db(left, right, dbname,
        #                     orig_disp_array1, orig_mask_array1, mappings)
        disp_arrays.append(orig_disp_array1)
        mask_arrays.append(orig_mask_array1)
        rect_params_list.append(rect_params_L)
        mappings_list.append(mappings)
    mse_exists = check_multi_stereo_entry(dbname, base_image)
    if not mse_exists:
        insert_multi_stereo(dbname, base_image, side_images)
    else:
        print("multi stereo with base image {} already in db".format(base_image))
    pc_in_db = check_point_cloud_entry(dbname, base_image)
    pc_path = os.path.join(TMP_PLY, "{}.ply".format(base_image[:-4]))
    pc_in_disk = os.path.exists(pc_path)
    if pc_in_db and pc_in_disk:
        print("point cloud of base image {} found both in disk and db".format(base_image))
    elif not pc_in_db and not pc_in_disk:
        print("point cloud of base image {} not found both in disk and db".format(base_image))
        print("triangulating multi stereo with base image {}...".format(base_image))
        points_3D, mask = triangulate_multi_db(base_image, side_images,
                                               disp_arrays, mask_arrays,
                                               mappings_list, rect_params_list,
                                               dbname)

        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))

        print("writing ply file...")
        # mask using inner-buffered overlap bounds
        # add mask based on occlusion/validity/RLCC

        masked_image = left["array"][mask==1]
        
        points_3D[:, 3:] = (masked_image.reshape(-1, 3)).astype(np.uint8)
        nviews = len(side_images) + 1
        write_ply_fpc_format(dbname, points_3D[:, 0:3], points_3D[:, 3:], base_image, nviews)
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))
        pc_id = insert_point_cloud_entry(dbname, base_image, points_3D[:, 0:3])
        update_pc_status_in_ms(dbname, base_image, pc_id)
    elif not pc_in_db and pc_in_disk:
        print("point cloud of base image {} found in disk but not in db".format(base_image))
        # plydata = storage.loadPly(pc_path)
        # points_3D = plydata["coords"]
        from plyfile import PlyData
        def simple_load_ply(infile):
            plydata = PlyData.read(infile)
            return plydata['vertex'].data # Returns a structured numpy array
        points_3D = simple_load_ply(pc_path)
        points_3D = np.stack([points_3D['x'], points_3D['y'], points_3D['z']], axis=1)
        pc_id = insert_point_cloud_entry(dbname, base_image, points_3D)
        update_pc_status_in_ms(dbname, base_image, pc_id)
    elif pc_in_db and not pc_in_disk:
        print("point cloud of base image {} found in db but not in disk".format(base_image))
        conn = psycopg2.connect(user="postgres",
                                password="postgres",
                                host="127.0.0.1",
                                port="5432",
                                database="{}".format(dbname))
        conn.autocommit = True
        # Create a cursor to perform database operations
        cur = conn.cursor()
        Q_SELECT_PC = """SELECT pc.pc_id
                         FROM point_cloud pc
                         LEFT JOIN multi_stereo ms
                           ON pc.pc_id=ms.pc_id
                         LEFT JOIN image im
                           ON ms.base_img_id=im.img_id
                         WHERE im.img_name='{base_image}';"""
        cur.execute(Q_SELECT_PC.format(base_image=base_image))
        pc_id = cur.fetchone()[0]
        Q_DELETE_PC_ENTRY = """DELETE FROM point_cloud
                               WHERE pc_id={pc_id};"""
        cur.execute(Q_DELETE_PC_ENTRY.format(pc_id=pc_id))
        cur.close()
        conn.close()
        print("triangulating multi stereo with base image {}...".format(base_image))
        points_3D, mask = triangulate_multi_db(base_image, side_images,
                                               disp_arrays, mask_arrays,
                                               mappings_list, rect_params_list,
                                               dbname)

        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))

        print("writing ply file...")
        # mask using inner-buffered overlap bounds
        # add mask based on occlusion/validity/RLCC

        masked_image = left["array"][mask==1]
        
        points_3D[:, 3:] = (masked_image.reshape(-1, 3)).astype(np.uint8)
        nviews = len(side_images) + 1
        write_ply_fpc_format(dbname, points_3D[:, 0:3], points_3D[:, 3:], base_image, nviews)
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("done in {:.2f} mins".format(t_mins))
        pc_id = insert_point_cloud_entry(dbname, base_image, points_3D[:, 0:3])
        update_pc_status_in_ms(dbname, base_image, pc_id)


def get_overlap(left_images, right_images, img_params):
    left_img = left_images[0]["fname"]
    right_img = right_images[0]["fname"]
    left_extent = img_params[left_img]["geobounds"]
    right_extent = img_params[right_img]["geobounds"]
    left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
    right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
    overlap = left_extent.intersection(right_extent)
    counter = 0
    for left, right in zip(left_images, right_images):
        if counter == 0:
            continue
        left_img = left["fname"]
        right_img = right["fname"]
        left_extent = img_params[left_img]["geobounds"]
        right_extent = img_params[right_img]["geobounds"]
        left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
        right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
        to_append = left_extent.intersection(right_extent)
        overlap = overlap.union(to_append)
        counter += 1
    return overlap


# def rasterize_PC(point_cloud, left_images, right_images, img_params, gsd, epsg):
#     # rasterize pointcloud using overlap of left and right images
#     print("rasterizing point cloud...")
#     overlap = get_overlap(left_images, right_images, img_params)
#     xmin, ymin, xmax, ymax = overlap.bounds
#     point_cloud = pd.read_csv(point_cloud, delimiter=" ",
#                               skiprows=10, names=["X", "Y", "Z", "R", "G", "B"],
#                               dtype=np.float64)
#     # coords = [(r["X"], r["Y"], r["Z"]) for _, r in point_cloud.iterrows()]
#     coords = [(r["X"], r["Y"], r["Z"]) for _, r in point_cloud.iterrows()\
#               if (xmin < r[0] < xmax) and (ymin < r[1] < ymax)]
#     # coords = [(r[0], r[1], r[2]) for r in point_cloud if (r[2] > 0) and (r[2] < 40)]
#     rec = np.recarray(len(coords), dtype=[("coords", float, 3)])
#     rec["coords"] = coords
#     proj = projection.Proj.from_epsg(int(epsg))
#     T = transformation.matrix(t=[xmin, ymax], s=[gsd, -gsd])
#     def agg_func(ids):
#         z = rec.coords[ids, 2]
#         n = len(ids)
#         # z = z.mean() if n > 0 else 0
#         z = np.median(z) if n > 0 else 0
#         return (n, z)
#     dtype = [("cell_count", int), ("z", float)]
#     ras = grid.voxelize(rec, T, agg_func=agg_func, dtype=dtype)
#     ras_rec = np.recarray(ras.shape, dtype=[("values", float)])
#     ras_rec["values"] = ras.z
#     ras = grid.Grid(proj, ras_rec, T)
#     outfilename = "tmp_ras_{}.tif".format(pair_idx)
#     outfile = os.path.join(TMP_DSM, outfilename)
#     storage.writeRaster(ras, outfile, field="values")
#     t = time.time()
#     t_mins = (t - T0) / 60.0
#     print("done in {:.2f} mins".format(t_mins))


if __name__=="__main__":
    T0 = time.time()
    DATAROOT = r"D:\Workspace\UT_postdoc\00_research\30_dataset\Use_Geo\Dataset-1"
    # # use downsampled images
    # IMGDIR = os.path.join(DATAROOT, "output_resized", "undistorted_images")
    # use full sized images
    IMGDIR = os.path.join(DATAROOT, "undistorted_images")
    # downsampling factor
    DF = 1
    # images are downsampled but parameters are not
    # # use downsampled images
    # DS_factor = 4
    # use full sized images
    DS_factor = 1
    # # use downsampled images
    # OVERLAP_BOUNDS_BUFFER = 200
    # use full sized images
    OVERLAP_BOUNDS_BUFFER = 300
    # TMP_ROOT = r'D:\\Workspace\\UT_postdoc\80_tools\\00_dronemappy\\dronemappy\\tmp'
    TMP_ROOT = r"D:\Workspace\UT_postdoc\40_experiments\03_use_geo\98_tmp_db"
    # TMP_ROOT = r"D:\Workspace\UT_postdoc\40_experiments\03_use_geo\97_tmp_db_finetune_ug1"
    # TMP_ROOT = r"D:\Workspace\UT_postdoc\40_experiments\03_use_geo\91_tmp_db_disp_shifting"
    TMP_PLY = os.path.join(TMP_ROOT, "ply")
    TMP_PLY_BOUNDS = os.path.join(TMP_PLY, "bounds")
    TMP_DISP = os.path.join(TMP_ROOT, "left_disp", "tmp_0")
    TMP_DISP_ORIG = os.path.join(TMP_ROOT, "left_disp", "orig")
    TMP_DISP_WARP = os.path.join(TMP_ROOT, "left_disp", "warp")
    TMP_DSM = os.path.join(TMP_ROOT, "dsm")
    TMP_IMG_LEFT = os.path.join(TMP_ROOT, "img", "left")
    TMP_IMG_RIGHT = os.path.join(TMP_ROOT, "img", "right")
    TMP_FNAME = "000000_05"
    TMP_RECT_FNAME_LIST = os.path.join(TMP_ROOT, "filenames.txt")
    TMP_LEFT_DISP = os.path.join(TMP_DISP, "{}.tif".format(TMP_FNAME))
    TMP_LEFT_PLY = os.path.join(TMP_PLY, "{}.ply".format(TMP_FNAME))
    SGM_CONFIG = os.path.join(TMP_ROOT, "sgm_config.json")
    SGM_OUT = os.path.join(TMP_ROOT, "left_disp", "sgm")
    SGM_DISP = os.path.join(SGM_OUT, "left_disparity.tif")
    SGM_MASK = os.path.join(SGM_OUT, "left_validity_mask.tif")
    LAS_FNAME = os.path.join(DATAROOT, "DIM_after_adjustment_dataset_C.las")
    # DISPARITY_METHOD = "SGM"
    DISPARITY_METHOD = "CNN"
    # MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\loss0-22_epoch34_ne45_ns15_bs12_lr1e-3_lambdas_1_1_1_0-25_0-25.pth.tar'
    MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\PASMnet_KITTI2015_epoch80.pth'
    # MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\epoch20_loss_0_19_epe_9_8_d3_22_3.pth.tar'
    # MODEL_PATH = 'D:\\Workspace\\UT_postdoc\\80_tools\\00_dronemappy\\dronemappy\\pretrained\\epoch79_loss_3_7_epe_5_8_d3_18_2.pth.tar'
    # # no masking
    # MASK_METHOD = None
    # # mask output from model
    # MASK_METHOD = "FM"
    # right-left consistency check mask
    MASK_METHOD = "RLCC"
    # disparity difference threshold, ignored when no masking is applied
    DISP_DIFF_THRESHOLD = 0.75
    # DISP_DIFF_THRESHOLD = 1.0
    # SUBSET_HEIGHT_OVERLAP = 256
    # SUBSET_WIDTH_OVERLAP = 640
    # SUBSET_WIDTH = 960
    # SUBSET_HEIGHT = 540
    # # use downsampled images
    # SUBSET_HEIGHT_OVERLAP = 384
    # SUBSET_WIDTH_OVERLAP = 960
    # SUBSET_WIDTH = 1440
    # SUBSET_HEIGHT = 810
    # use full sized images
    SUBSET_HEIGHT_OVERLAP = 512
    SUBSET_WIDTH_OVERLAP = 1280
    SUBSET_WIDTH = 2880
    SUBSET_HEIGHT = 1620
    # SUBSET_HEIGHT_OVERLAP = 512
    # SUBSET_WIDTH_OVERLAP = 1280
    # SUBSET_WIDTH = 4320
    # SUBSET_HEIGHT = 2430

    # on full res, shift disparity map
    # based on principal point and mean flying height
    # DISPARITY_SHIFT_RATIO = 0.7
    DISPARITY_SHIFT_RATIO = 0.9

    # import laspy
    # las = laspy.read(LAS_FNAME)
    # print(np.array(las.z).mean())

    EPSG = 32632

    # POINTS_DB_NAME = "usegeo_dataset1"
    POINTS_DB_NAME = "tmp_finetune_ug1"

    # N_PAIRS = 10

    # N_PAIRS = 1
    # # shift = 0
    # # imagefilenames = [i for i in sorted(os.listdir(IMGDIR)) if i.endswith("jpg")]
    # # imagefilenames = imagefilenames[:N_PAIRS+1]
    # # flip image pairs
    # # _, __ = imagefilenames[0], imagefilenames[1]
    # # imagefilenames[0] = __
    # # imagefilenames[1] = _
    # # base_image = "2021-04-23_13-17-14_S2223314_DxO.jpg"
    # # base_image = "2021-04-23_13-17-20_S2223314_DxO.jpg"
    # base_image = "2021-04-23_13-17-22_S2223314_DxO.jpg"
    # # side_image = "2021-04-23_13-17-12_S2223314_DxO.jpg"
    # # side_image = "2021-04-23_13-17-16_S2223314_DxO.jpg"
    # # side_image = "2021-04-23_13-17-20_S2223314_DxO.jpg"
    # # side_image = "2021-04-23_13-17-22_S2223314_DxO.jpg"
    # side_image = "2021-04-23_13-17-24_S2223314_DxO.jpg"
    # imagefilenames = [base_image, side_image]
    # n_images = len(imagefilenames)
    # img_objects = OrderedDict()
    # img_arrays = OrderedDict()
    # print("reading and downsampling {} images...".format(n_images))
    # for i, imgfname in enumerate(imagefilenames):
    #     imgpath = os.path.join(IMGDIR, imgfname)
    #     img = gdal.Open(imgpath)
    #     img_objects[imgfname] = img
    #     imgarray = img.ReadAsArray()
    #     nbands, nrows, ncols = imgarray.shape
    #     output_shape = (nbands, int(nrows/DF), int(ncols/DF))
    #     downsampled = np.zeros((output_shape))
    #     downsampled = transform.resize(imgarray, output_shape)
    #     downsampled = np.transpose(downsampled, [1, 2, 0])
    #     img_arrays[imgfname] = downsampled

    # t = time.time()
    # t_mins = (t - T0) / 60.0
    # print("done in {:.2f} mins".format(t_mins))

    # metric_calib = os.path.join(DATAROOT, "image_orientations.xyz")
    # img_params = init_img_params(metric_calib, IMGDIR, LAS_FNAME, DS_factor, T0)
    # GSD = estimate_GSD(img_params, DF)

    # # # reconstruct stereopair wise
    # # left_images = []
    # # right_images = []
    # # for i in range(N_PAIRS):
    # #     LEFT_IMG = imagefilenames[i]
    # #     RIGHT_IMG = imagefilenames[i + 1]
    # #     pair_idx = i
    # #     LEFT_IMG_ARRAY = img_arrays[LEFT_IMG]
    # #     RIGHT_IMG_ARRAY = img_arrays[RIGHT_IMG]
    # #     LEFT_IMG_OBJECT = img_objects[LEFT_IMG]
    # #     RIGHT_IMG_OBJECT = img_objects[RIGHT_IMG]
    # #     LEFT = {"fname": LEFT_IMG,
    # #             "object": LEFT_IMG_OBJECT,
    # #             "array": LEFT_IMG_ARRAY}
    # #     RIGHT = {"fname": RIGHT_IMG,
    # #              "object": RIGHT_IMG_OBJECT,
    # #              "array": RIGHT_IMG_ARRAY}
    # #     left_images.append(LEFT)
    # #     right_images.append(RIGHT)
    # #     point_cloud = stereo_reconstruct(LEFT, RIGHT, img_params, DF, DISPARITY_METHOD,
    # #                                      MASK_METHOD, DISP_DIFF_THRESHOLD, pair_idx)
    # #     # point_cloud = TMP_LEFT_PLY


    # # N_PAIRS = 5
    # # shift = 0
    # # imagefilenames = [i for i in sorted(os.listdir(IMGDIR)) if i.endswith("jpg")]
    # # N_PAIRS = len(imagefilenames) - 1
    # # imagefilenames = imagefilenames[:N_PAIRS+1]
    # # # flip image pairs
    # # _, __ = imagefilenames[0], imagefilenames[1]
    # # imagefilenames[0] = __
    # # imagefilenames[1] = _
    
    # # test specific multi-stereo
    # # imagefilenames = ["2021-04-23_13-17-20_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-22_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-24_S2223314_DxO.jpg"]
    # imagefilenames = ["2021-04-23_13-17-22_S2223314_DxO.jpg",
    #                   "2021-04-23_13-17-24_S2223314_DxO.jpg",
    #                   "2021-04-23_13-17-26_S2223314_DxO.jpg"]
    # n_images = len(imagefilenames)
    # img_objects = OrderedDict()
    # img_arrays = OrderedDict()
    # # print("reading and downsampling {} images...".format(n_images))
    # # for i, imgfname in enumerate(imagefilenames):
    # #     imgpath = os.path.join(IMGDIR, imgfname)
    # #     img = gdal.Open(imgpath)
    # #     img_objects[imgfname] = img
    # #     imgarray = img.ReadAsArray()
    # #     img_arrays[imgfname] =imgarray
    #     # nbands, nrows, ncols = imgarray.shape
    #     # output_shape = (nbands, int(nrows/DF), int(ncols/DF))
    #     # downsampled = np.zeros((output_shape))
    #     # downsampled = transform.resize(imgarray, output_shape)
    #     # downsampled = np.transpose(downsampled, [1, 2, 0])
    #     # img_arrays[imgfname] = downsampled

    # # t = time.time()
    # # t_mins = (t - T0) / 60.0
    # # print("done in {:.2f} mins".format(t_mins))

    # # reconstruct multi stereo wise
    # min_overlap = 0.7
    # test_img_idx = 1
    # multiview_dict = {}
    # side_counter = 0
    # side_images = []
    # pair_idx = 0
    # BASE_IMG = imagefilenames[test_img_idx]
    # side_images = [imagefilenames[0], imagefilenames[2]]
    # # BASE_IMG_ARRAY = img_arrays[BASE_IMG]
    # # BASE_IMG_OBJECT = img_objects[BASE_IMG]
    # # for fname in imagefilenames:
    # #     if fname == BASE_IMG:
    # #         continue
    # #     left_extent = img_params[BASE_IMG]["geobounds"]
    # #     right_extent = img_params[fname]["geobounds"]
    # #     left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
    # #     right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
    # #     overlap = left_extent.intersection(right_extent)
    # #     overlap_ratio = overlap.area / left_extent.area
    # #     if overlap_ratio > min_overlap:
    # #         side_counter += 1
    # #         side_images.append(fname)

    # multiview_dict[BASE_IMG] = side_images
    # # only uses right-left consistency mask
    # point_cloud = mvs_reconstruct(multiview_dict, img_params, DF, DISPARITY_METHOD,
    #                               DISP_DIFF_THRESHOLD, pair_idx)


    # # base_images = []
    # # side_images = []
    # # for i in range(N_PAIRS):
    # #     BASE_IMG = imagefilenames[i]
    # #     # BASE_IMG_ARRAY = img_arrays[BASE_IMG]
    # #     BASE_IMG_OBJECT = img_objects[BASE_IMG]
    # #     side_counter = 0
    # #     for fname in imagefilenames:
    # #         if fname == BASE_IMG:
    # #             continue
    # #         left_extent = img_params[BASE_IMG]["geobounds"]
    # #         right_extent = img_params[fname]["geobounds"]
    # #         left_extent = Polygon([tuple(i[:-1]) for i in left_extent])
    # #         right_extent = Polygon([tuple(i[:-1]) for i in right_extent])
    # #         overlap = left_extent.intersection(right_extent)
    # #         overlap_ratio = overlap.area / left_extent.area
    # #         if overlap_ratio > min_overlap:
    # #             side_counter += 1
    # #     print("{} has {} side  images".format(BASE_IMG, side_counter))


    # # reconstruct multi stereo wise using db
    # init_DB(POINTS_DB_NAME,
    #         DATAROOT, IMGDIR, LAS_FNAME,
    #         DS_factor, T0, DF, EPSG)
    # imagefilenames = ["2021-04-23_13-17-20_S2223314_DxO.jpg",
    #                   "2021-04-23_13-17-22_S2223314_DxO.jpg",
    #                   "2021-04-23_13-17-24_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-22_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-24_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-26_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-24_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-26_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-28_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-26_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-28_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-30_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-28_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-30_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-32_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-30_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-32_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-34_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-32_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-34_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-36_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-34_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-36_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-38_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-36_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-38_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-40_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-38_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-40_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-42_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-40_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-42_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-44_S2223314_DxO.jpg"]
    # # imagefilenames = ["2021-04-23_13-17-42_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-44_S2223314_DxO.jpg",
    # #                   "2021-04-23_13-17-46_S2223314_DxO.jpg"]
    # test_img_idx = 1
    # multiview_dict = {}
    # side_images = []
    # pair_idx = 0
    # BASE_IMG = imagefilenames[test_img_idx]
    # side_images = [imagefilenames[0], imagefilenames[2]]
    # multiview_dict[BASE_IMG] = side_images
    # # only uses right-left consistency mask
    # point_cloud = mvs_reconstruct_db(multiview_dict, DF, DISPARITY_METHOD,
    #                                  DISP_DIFF_THRESHOLD, POINTS_DB_NAME)

    # min_overlap = 0.7
    # init_DB(POINTS_DB_NAME,
    #         DATAROOT, IMGDIR, LAS_FNAME,
    #         DS_factor, T0, DF, EPSG)
    # imagefilenames = get_image_filenames(POINTS_DB_NAME)
    # multiview_dicts = []
    # base_counter = 0
    # for base_image in imagefilenames:
    #     side_counter = 0
    #     side_images = []
    #     overlap_ratios = []
    #     for side_image in imagefilenames:
    #         if side_image == base_image:
    #             continue
    #         base_extent = get_img_params_db(base_image, POINTS_DB_NAME)["geobounds"]
    #         side_extent = get_img_params_db(side_image, POINTS_DB_NAME)["geobounds"]
    #         base_extent = Polygon([tuple(i[:-1]) for i in base_extent])
    #         side_extent = Polygon([tuple(i[:-1]) for i in side_extent])
    #         overlap = base_extent.intersection(side_extent)
    #         overlap_ratio = overlap.area / base_extent.area
    #         if overlap_ratio > min_overlap:
    #             side_counter += 1
    #             side_images.append(side_image)
    #             overlap_ratios.append(overlap_ratio)
    #     import heapq
    #     if len(side_images) > 1:
    #         largest_overlap_ratios = heapq.nlargest(2, overlap_ratios)
    #         side_image1 = side_images[overlap_ratios.index(largest_overlap_ratios[0])]
    #         side_image2 = side_images[overlap_ratios.index(largest_overlap_ratios[1])]
    #         filtered_side_images = [side_image1, side_image2]
    #     else:
    #         filtered_side_images = side_images
    #     # print(side_images)
    #     # print(overlap_ratios)
    #     # print(filtered_side_images)
    #     multiview_dict = {}
    #     multiview_dict[base_image] = filtered_side_images
    #     multiview_dicts.append(multiview_dict)
    #     t = time.time()
    #     t_mins = (t - T0) / 60.0
    #     print("finished preprocessing {} multi views in {:.2f} mins".format(base_counter+1, t_mins))
    #     base_counter += 1

    # import pickle
    # with open('multiview_dicts.list', 'wb') as f:
    #     pickle.dump(multiview_dicts, f)

    init_DB(POINTS_DB_NAME,
            DATAROOT, IMGDIR, LAS_FNAME,
            DS_factor, T0, DF, EPSG)
    # priority_fnames_file = r"D:\Workspace\UT_postdoc\40_experiments\03_use_geo\98_tmp_db\sample_overlapping_bounds_fnames.csv"
    # with open(priority_fnames_file) as f:
    #     priority_fnames = [line.rstrip() for line in f]
    multiview_dict_path = r"D:\Workspace\UT_postdoc\40_experiments\03_use_geo\98_tmp_db\multiview_dicts.list"
    with open(multiview_dict_path, "rb") as f:
        multiview_dicts = pickle.load(f)
    base_counter = 0
    for multiview_dict in multiview_dicts:
        base_image = list(multiview_dict.keys())[0]
        # if base_image not in priority_fnames:
        # if base_image in priority_fnames:
        # if base_image != "2021-04-23_13-23-29_S2223314_DxO.jpg":
        if base_image != "2021-04-23_13-17-22_S2223314_DxO.jpg":
            continue
        point_cloud = mvs_reconstruct_db(multiview_dict, DF, DISPARITY_METHOD,
                                         DISP_DIFF_THRESHOLD, POINTS_DB_NAME)
        t = time.time()
        t_mins = (t - T0) / 60.0
        print("finished preprocessing {} multi views in {:.2f} mins".format(base_counter+1, t_mins))
        base_counter += 1

    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

# rasterized_PC = rasterize_PC(point_cloud, left_images, right_images, img_params, GSD, EPSG)

# search_criteria = "tmp_ras*.tif"
# q = os.path.join(TMP_DSM, search_criteria)
# dsm_fps = glob.glob(q)
# src_files_to_mosaic = []
# for fp in dsm_fps:
#     src = rasterio.open(fp)
#     src_files_to_mosaic.append(src)
# res = Resampling.average
# mosaic, out_trans = merge(src_files_to_mosaic, nodata=0, resampling=res)
# # show(mosaic)
# out_meta = src.meta.copy()
# out_meta.update({'nodata': 0,
#                  'width': mosaic.shape[2],
#                  'height': mosaic.shape[1],
#                  'transform': out_trans})
# out_fname = "merged_overlaps.tif"
# out_fp = os.path.join(TMP_DSM, out_fname)
# with rasterio.open(out_fp, "w", **out_meta) as dest:
#     dest.write(mosaic)
# with rasterio.open(out_fp, "r") as ds:
#     polys = list(dataset_features(ds, bidx=1, as_mask=True, geographic=False, band=False))
#     multi_poly = MultiPolygon([shape(poly["geometry"]) for poly in polys])
#     merged_bounds = multi_poly.convex_hull
#     merged_mask = rasterio.features.rasterize([merged_bounds],
#                                               out_shape=mosaic.shape[1:],
#                                               transform=out_trans)
# merged_mask = np.expand_dims(merged_mask, axis=0)
# mosaic = fillnodata(mosaic, (mosaic!=0))
# mosaic[merged_mask==0] = 0
# out_fname = "merged_overlaps_filled.tif"
# out_fp = os.path.join(TMP_DSM, out_fname)
# with rasterio.open(out_fp, "w", **out_meta) as dest:
#     dest.write(mosaic)
