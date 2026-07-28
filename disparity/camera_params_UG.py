import time
import math
import os
from collections import OrderedDict
import numpy as np
import pandas as pd
from osgeo import gdal
import laspy


def get_mean_surface_elevation(las_fname):
    las = laspy.read(las_fname)
    mean_elev = np.array(las.z).mean()
    return mean_elev


def add_geobounds(img_params, mean_flying_height, mean_gcp_height):
    # add extent of images
    z_f = mean_flying_height
    for img, params in img_params.items():
        K = img_params[img]["camera_matrix"]
        R = img_params[img]["rotation_matrix"]
        C = np.array([[img_params[img]["X"]],
                      [img_params[img]["Y"]],
                      [img_params[img]["Z"]]], dtype=np.float64)
        topleft_x = -K[0][2]
        topleft_y = -K[1][2]
        topright_x = img_params[img]["ncols"] - K[0][2]
        topright_y = -K[1][2]
        botright_x = img_params[img]["ncols"] - K[0][2]
        botright_y = img_params[img]["nrows"] - K[1][2]
        botleft_x = -K[0][2]
        botleft_y = img_params[img]["nrows"] - K[1][2]
        # from topleft then clockwise
        corner_coords = [(topleft_x, topleft_y),
                         (topright_x, topright_y),
                         (botright_x, botright_y),
                         (botleft_x, botleft_y)]
        f_x = K[0][0]
        f_y = K[1][1]
        corner_coords_delta = [np.array([[(x*z_f/f_x)],
                                        [(y*z_f/f_y)],
                                        [z_f]])
                               for x, y in corner_coords]
        corner_coords_world = [(np.linalg.inv(R) @ CCD) + C
                               for CCD in corner_coords_delta]
        img_params[img]["geobounds"] = corner_coords_world
        img_params[img]["mean_gcp_height"] = mean_gcp_height
        img_params[img]["mean_flying_height"] = mean_flying_height
    return img_params


def fetch_camera_params(metric_calib, imgdir, DS_factor):
    # fetch camera and image parameters
    df = pd.read_csv(metric_calib, delim_whitespace=True)
    img_params = OrderedDict()
    for index, row in df.iterrows():
        imgfname = row["#label"]
        # # use downsampled images
        # imgfname = imgfname[:-4] + "_res.jpg"
        imgpath = os.path.join(imgdir, imgfname)
        img = gdal.Open(imgpath)
        imgarray = img.ReadAsArray()
        nbands, nrows, ncols = imgarray.shape
        X = row["X0"]
        Y = row["Y0"]
        Z = row["Z0"]
        focal_length = row["c"]
        projection_center_X = row["x0"]
        projection_center_Y = row["y0"]
        kappa = np.radians(row["kappa[deg]"])
        phi = np.radians(row["phi[deg]"])
        omega = np.radians(row["omega[deg]"])
        camera_matrix_K = np.zeros(shape=(3, 3), dtype=np.float64)
        camera_matrix_K[0][0] = focal_length / DS_factor
        camera_matrix_K[1][1] = focal_length / DS_factor
        camera_matrix_K[2][2] = 1
        camera_matrix_K[0][2] = projection_center_X / DS_factor
        camera_matrix_K[1][2] = -projection_center_Y / DS_factor
        R = np.zeros(shape=(3, 3), dtype=np.float64)
        R[0,0]=(np.cos(phi))*(np.cos(kappa))
        R[0,1]=-(np.cos(phi)*np.sin(kappa))
        R[0,2]=np.sin(phi)
        R[1,0]=(np.cos(omega)*np.sin(kappa))+(np.sin(omega)*np.sin(phi)*np.cos(kappa))
        R[1,1]=(np.cos(omega)*np.cos(kappa))-(np.sin(omega)*np.sin(phi)*np.sin(kappa))
        R[1,2]=(-np.sin(omega)*np.cos(phi))
        R[2,0]=(np.sin(omega)*np.sin(kappa)-np.cos(omega)*np.sin(phi)*np.cos(kappa))
        R[2,1]=(np.sin(omega)*np.cos(kappa)+np.cos(omega)*np.sin(phi)*np.sin(kappa))
        R[2,2]=(np.cos(omega)*np.cos(phi))
        R = R.T
        R[1] = -R[1]
        R[2] = -R[2]
        camera_rotation_R = R
        # R = camera_rotation_R
        # camera_rotation_R = np.linalg.det(R) ** (1/3) * R
        radial_distortion = np.array([row["a3"], row["a4"], row["a5"], row["a6"]])
        tangential_distortion = np.array([row["rho0"]])
        img_params[imgfname] = {"X": X,
                                "Y": Y,
                                "Z": Z,
                                "nrows": nrows,
                                "ncols": ncols,
                                "camera_matrix": camera_matrix_K,
                                "rotation_matrix": camera_rotation_R,
                                "radial_distortion": radial_distortion,
                                "tangential_distortion": tangential_distortion}
    # print(img_params)
    return img_params


def init_img_params(metric_calib, imgdir, las_fname, DS_factor, T0):
    # initialize image and camera parameters
    print("collecting camera and image parameters...")
    img_params = fetch_camera_params(metric_calib, imgdir, DS_factor)
    img_Zs = [img_params[i]["Z"] for i, _ in img_params.items()]
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    print("estimating image footprints...")
    mean_gcp_height = get_mean_surface_elevation(las_fname)
    mean_flying_height = np.array(img_Zs).mean() - mean_gcp_height
    img_params = add_geobounds(img_params, mean_flying_height, mean_gcp_height)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))
    return img_params


def estimate_GSD(img_params, DF):
    # estimate ground sampling distance based on
    # image ground footprint and number of pixels
    samplekey = list(img_params.keys())[0]
    extent = img_params[samplekey]["geobounds"]
    nrows = img_params[samplekey]["nrows"]
    ncols = img_params[samplekey]["ncols"]
    ncols_metric = math.sqrt((extent[0][0] - extent[1][0]) ** 2 +
                             (extent[0][1] - extent[1][1]) ** 2)
    nrows_metric = math.sqrt((extent[0][0] - extent[-1][0]) ** 2 +
                             (extent[0][1] - extent[-1][1]) ** 2)
    gsd_ncols = ncols_metric / float(ncols)
    gsd_nrows = nrows_metric / float(nrows)
    gsd = (gsd_ncols + gsd_nrows) / 2.0
    print("GSD: {}".format(gsd))
    return gsd * DF