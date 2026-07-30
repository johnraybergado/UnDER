"""
Builds per-image camera parameter dictionaries (position, rotation,
camera matrix, distortion) from the UseGeo metric-calibration file, plus
derived geographic footprints (geobounds) and GSD estimation.

Boundary: this is where raw calibration/orientation data becomes the
CameraParams dicts every other module consumes (rectify.py, stereo.py,
db_core.py's image_param table). Runs once at DB-init time
(db_core.init_db); nothing here talks to the database directly.
"""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict
from typing import TypedDict, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from osgeo import gdal  # pyright: ignore[reportMissingTypeStubs]
import laspy  # pyright: ignore[reportMissingTypeStubs]


# ---------------------------------------------------------------------------
# TypedDicts
# ---------------------------------------------------------------------------

class CameraParams(TypedDict):
    X: float
    Y: float
    Z: float
    nrows: int
    ncols: int
    camera_matrix: npt.NDArray[np.float64]
    rotation_matrix: npt.NDArray[np.float64]
    radial_distortion: npt.NDArray[np.float64]
    tangential_distortion: npt.NDArray[np.float64]


class CameraParamsWithGeo(CameraParams, total=False):
    geobounds: list[npt.NDArray[np.float64]]
    mean_gcp_height: float
    mean_flying_height: float


ImgParamsDict = OrderedDict[str, CameraParamsWithGeo]


# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------

def get_mean_surface_elevation(las_fname: str) -> float:
    """Return the mean Z elevation from a LAS file."""
    las = laspy.read(las_fname)  # pyright: ignore[reportUnknownMemberType]
    mean_elev: float = float(np.array(las.z).mean())
    return mean_elev


def add_geobounds(
    img_params: ImgParamsDict,
    mean_flying_height: float,
    mean_gcp_height: float,
) -> ImgParamsDict:
    """Add geographic extent (footprint) and flight metadata to each image entry."""
    z_f: float = mean_flying_height
    for img in img_params:
        K: npt.NDArray[np.float64] = img_params[img]["camera_matrix"]
        R: npt.NDArray[np.float64] = img_params[img]["rotation_matrix"]
        C: npt.NDArray[np.float64] = np.array(
            [
                [img_params[img]["X"]],
                [img_params[img]["Y"]],
                [img_params[img]["Z"]],
            ],
            dtype=np.float64,
        )

        topleft_x:  float = -float(K[0][2])
        topleft_y:  float = -float(K[1][2])
        topright_x: float = float(img_params[img]["ncols"]) - float(K[0][2])
        topright_y: float = -float(K[1][2])
        botright_x: float = float(img_params[img]["ncols"]) - float(K[0][2])
        botright_y: float = float(img_params[img]["nrows"]) - float(K[1][2])
        botleft_x:  float = -float(K[0][2])
        botleft_y:  float = float(img_params[img]["nrows"]) - float(K[1][2])

        # from top-left then clockwise
        corner_coords: list[tuple[float, float]] = [
            (topleft_x,  topleft_y),
            (topright_x, topright_y),
            (botright_x, botright_y),
            (botleft_x,  botleft_y),
        ]

        f_x: float = float(K[0][0])
        f_y: float = float(K[1][1])

        corner_coords_delta: list[npt.NDArray[np.float64]] = [
            np.array(
                [[(x * z_f / f_x)], [(y * z_f / f_y)], [z_f]],
                dtype=np.float64,
            )
            for x, y in corner_coords
        ]

        corner_coords_world: list[npt.NDArray[np.float64]] = [
            (np.linalg.inv(R) @ ccd) + C
            for ccd in corner_coords_delta
        ]

        img_params[img]["geobounds"] = corner_coords_world
        img_params[img]["mean_gcp_height"] = mean_gcp_height
        img_params[img]["mean_flying_height"] = mean_flying_height

    return img_params


def fetch_camera_params(
    metric_calib: str,
    imgdir: str,
    DS_factor: float,
) -> ImgParamsDict:
    """Read metric calibration CSV and build per-image camera parameter dicts."""
    df: pd.DataFrame = pd.read_csv(metric_calib, sep=r"\s+")
    img_params: ImgParamsDict = OrderedDict()

    for _, row in df.iterrows():
        imgfname: str = str(row["#label"])
        imgpath: str = os.path.join(imgdir, imgfname)

        ds = cast(gdal.Dataset, gdal.Open(imgpath))  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
        imgarray: npt.NDArray[np.float64] = cast(
            npt.NDArray[np.float64],
            ds.ReadAsArray(),  # pyright: ignore[reportUnknownMemberType]
        )
        nrows: int
        ncols: int
        _, nrows, ncols = imgarray.shape

        X:    float = float(row["X0"])
        Y:    float = float(row["Y0"])
        Z:    float = float(row["Z0"])
        focal_length:        float = float(row["c"])
        projection_center_X: float = float(row["x0"])
        projection_center_Y: float = float(row["y0"])
        kappa: float = np.radians(float(row["kappa[deg]"]))
        phi:   float = np.radians(float(row["phi[deg]"]))
        omega: float = np.radians(float(row["omega[deg]"]))

        camera_matrix_K: npt.NDArray[np.float64] = np.zeros(
            shape=(3, 3), dtype=np.float64
        )
        camera_matrix_K[0][0] = focal_length / DS_factor
        camera_matrix_K[1][1] = focal_length / DS_factor
        camera_matrix_K[2][2] = 1.0
        camera_matrix_K[0][2] = projection_center_X / DS_factor
        camera_matrix_K[1][2] = -projection_center_Y / DS_factor

        R: npt.NDArray[np.float64] = np.zeros(shape=(3, 3), dtype=np.float64)
        R[0, 0] = np.cos(phi) * np.cos(kappa)
        R[0, 1] = -(np.cos(phi) * np.sin(kappa))
        R[0, 2] = np.sin(phi)
        R[1, 0] = (np.cos(omega) * np.sin(kappa)) + (np.sin(omega) * np.sin(phi) * np.cos(kappa))
        R[1, 1] = (np.cos(omega) * np.cos(kappa)) - (np.sin(omega) * np.sin(phi) * np.sin(kappa))
        R[1, 2] = -(np.sin(omega) * np.cos(phi))
        R[2, 0] = np.sin(omega) * np.sin(kappa) - np.cos(omega) * np.sin(phi) * np.cos(kappa)
        R[2, 1] = np.sin(omega) * np.cos(kappa) + np.cos(omega) * np.sin(phi) * np.sin(kappa)
        R[2, 2] = np.cos(omega) * np.cos(phi)
        R_t = R.T
        R_t[1] = -R_t[1]
        R_t[2] = -R_t[2]
        camera_rotation_R: npt.NDArray[np.float64] = R_t

        radial_distortion: npt.NDArray[np.float64] = np.array(
            [float(row["a3"]), float(row["a4"]), float(row["a5"]), float(row["a6"])],
            dtype=np.float64,
        )
        tangential_distortion: npt.NDArray[np.float64] = np.array(
            [float(row["rho0"])],
            dtype=np.float64,
        )

        img_params[imgfname] = {
            "X": X,
            "Y": Y,
            "Z": Z,
            "nrows": nrows,
            "ncols": ncols,
            "camera_matrix": camera_matrix_K,
            "rotation_matrix": camera_rotation_R,
            "radial_distortion": radial_distortion,
            "tangential_distortion": tangential_distortion,
        }

    return img_params


def init_img_params(
    metric_calib: str,
    imgdir: str,
    las_fname: str,
    DS_factor: float,
    T0: float,
) -> ImgParamsDict:
    """Initialize image and camera parameters, then attach geographic footprints."""
    print("collecting camera and image parameters...")
    img_params: ImgParamsDict = fetch_camera_params(metric_calib, imgdir, DS_factor)
    img_Zs: list[float] = [img_params[i]["Z"] for i in img_params]
    t: float = time.time()
    t_mins: float = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))

    print("estimating image footprints...")
    mean_gcp_height: float = get_mean_surface_elevation(las_fname)
    mean_flying_height: float = float(np.array(img_Zs).mean()) - mean_gcp_height
    img_params = add_geobounds(img_params, mean_flying_height, mean_gcp_height)
    t = time.time()
    t_mins = (t - T0) / 60.0
    print("done in {:.2f} mins".format(t_mins))
    return img_params


def estimate_GSD(img_params: ImgParamsDict) -> float:
    """Estimate ground sampling distance from image footprint and pixel count."""
    samplekey: str = list(img_params.keys())[0]
    extent: list[npt.NDArray[np.float64]] = img_params[samplekey]["geobounds"]  # type: ignore[typeddict-item]
    nrows: int = img_params[samplekey]["nrows"]
    ncols: int = img_params[samplekey]["ncols"]

    ncols_metric: float = math.sqrt(
        float((extent[0][0] - extent[1][0]) ** 2 + (extent[0][1] - extent[1][1]) ** 2)
    )
    nrows_metric: float = math.sqrt(
        float((extent[0][0] - extent[-1][0]) ** 2 + (extent[0][1] - extent[-1][1]) ** 2)
    )
    gsd_ncols: float = ncols_metric / float(ncols)
    gsd_nrows: float = nrows_metric / float(nrows)
    gsd: float = (gsd_ncols + gsd_nrows) / 2.0
    print("GSD: {}".format(gsd))
    return gsd
