from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import numpy.typing as npt
from skimage import transform


class CameraParams(TypedDict):
    rotation_matrix: npt.NDArray[np.float64]
    camera_matrix: npt.NDArray[np.float64]
    X: float
    Y: float
    Z: float


class RectificationResult(TypedDict):
    image_pairs: list[npt.NDArray[np.float64]]
    rect_params: RectParams


class RectParams(TypedDict):
    H1: npt.NDArray[np.float64]
    H2: npt.NDArray[np.float64]
    H_shift: npt.NDArray[np.float64]
    K: npt.NDArray[np.float64]
    R: npt.NDArray[np.float64]


def redefine_bounds(
    imgs: list[npt.NDArray[np.float64]],
    H: list[npt.NDArray[np.float64]],
# ) -> tuple[tuple[float, float, float, float], tuple[int, int, int, int]]:
) -> tuple[float, float, float, float]:
    """
    Calculate new bounds such that all pixels within the image pairs
    have positive pixel coordinates.

    :param imgs: list of image arrays
    :param H: list of homography matrices
    :return: bounds (x_min, x_max, y_min, y_max) and their image indices
    """
    x_min: float = np.inf
    y_min: float = np.inf
    x_max: float = -np.inf
    y_max: float = -np.inf
    # x_min_idx: int = 0
    # y_min_idx: int = 0
    # x_max_idx: int = 0
    # y_max_idx: int = 0

    for idx, img in enumerate(imgs):
        nrows1: int = img.shape[0]
        ncols1: int = img.shape[1]

        c: float = 1.0
        x_corners: list[npt.NDArray[np.float64]] = [
            np.array([0,      0,      c], dtype=np.float64),  # top-left
            np.array([0,      nrows1, c], dtype=np.float64),  # top-right
            np.array([ncols1, 0,      c], dtype=np.float64),  # bottom-left
            np.array([ncols1, nrows1, c], dtype=np.float64),  # bottom-right
        ]

        H_inv: npt.NDArray[np.float64] = np.linalg.inv(H[idx])
        for x_corner in x_corners:
            corner: npt.NDArray[np.float64] = H_inv @ x_corner
            norm_corner: npt.NDArray[np.float64] = corner / corner[-1]
            x: float = float(norm_corner[0])
            y: float = float(norm_corner[1])
            # if x < x_min:
            #     x_min = x
            #     x_min_idx = idx
            # if y < y_min:
            #     y_min = y
            #     y_min_idx = idx
            # if x > x_max:
            #     x_max = x
            #     x_max_idx = idx
            # if y > y_max:
            #     y_max = y
            #     y_max_idx = idx
            if x < x_min:
                x_min = x
            if y < y_min:
                y_min = y
            if x > x_max:
                x_max = x
            if y > y_max:
                y_max = y

    bounds: tuple[float, float, float, float] = (x_min, x_max, y_min, y_max)
    # bounds_idx: tuple[int, int, int, int] = (x_min_idx, x_max_idx, y_min_idx, y_max_idx)
    # return bounds, bounds_idx
    return bounds


def rectify_stereopair(
    imgarray1: npt.NDArray[np.float64],
    imgarray2: npt.NDArray[np.float64],
    imgparams1: CameraParams,
    imgparams2: CameraParams,
    DF: float,
) -> RectificationResult:
    """
    Rectify two downsampled image pairs.

    :param imgarray1: numpy array of the left image
    :param imgarray2: numpy array of the right image
    :param imgparams1: camera parameters of the left image
    :param imgparams2: camera parameters of the right image
    :param DF: downsampling factor constant
    :return: transformed image arrays and rectification parameters
    """
    R1: npt.NDArray[np.float64] = imgparams1["rotation_matrix"]
    R2: npt.NDArray[np.float64] = imgparams2["rotation_matrix"]
    Z1: npt.NDArray[np.float64] = np.array(
        [imgparams1["X"], imgparams1["Y"], imgparams1["Z"]], dtype=np.float64
    )
    Z2: npt.NDArray[np.float64] = np.array(
        [imgparams2["X"], imgparams2["Y"], imgparams2["Z"]], dtype=np.float64
    )

    # same camera matrix for all images — copy to avoid mutating the original
    K: npt.NDArray[np.float64] = imgparams1["camera_matrix"].copy()
    K_rescaled = K / DF
    K_rescaled[-1, -1] = K_rescaled[-1, -1] * DF

    R: npt.NDArray[np.float64] = np.zeros(R1.shape, dtype=np.float64)
    B_2_1: npt.NDArray[np.float64] = Z2 - Z1
    R[0] = B_2_1 / np.linalg.norm(B_2_1)

    C1: npt.NDArray[np.float64] = np.concatenate(
        (np.eye(3, dtype=np.float64), -np.expand_dims(Z1, axis=1)), axis=1
    )
    C2: npt.NDArray[np.float64] = np.concatenate(
        (np.eye(3, dtype=np.float64), -np.expand_dims(Z2, axis=1)), axis=1
    )

    P1: npt.NDArray[np.float64] = K @ R1 @ C1
    P2: npt.NDArray[np.float64] = K @ R2 @ C2

    d1: npt.NDArray[np.float64] = -P1[2, :3]
    d2: npt.NDArray[np.float64] = -P2[2, :3]

    d_mean: npt.NDArray[np.float64] = (
        d1 / np.linalg.norm(d1)
    ) + (
        d2 / np.linalg.norm(d2)
    )
    d_mean = d_mean / np.linalg.norm(d_mean)

    d_y: npt.NDArray[np.float64] = np.cross(B_2_1, d_mean)
    d_z: npt.NDArray[np.float64] = np.cross(B_2_1, d_y)
    R[1] = d_y / np.linalg.norm(d_y)
    R[2] = d_z / np.linalg.norm(d_z)

    H1: npt.NDArray[np.float64] = K @ R1 @ R.T @ np.linalg.inv(K)
    H2: npt.NDArray[np.float64] = K @ R2 @ R.T @ np.linalg.inv(K)

    # homographies to warp image pair to a stereo normal pair
    H: list[npt.NDArray[np.float64]] = [H1, H2]
    # b, b_idx = redefine_bounds([imgarray1, imgarray2], H)
    b = redefine_bounds([imgarray1, imgarray2], H)

    outshape: tuple[int, int] = (math.ceil(b[3] - b[2]), math.ceil(b[1] - b[0]))
    shifts: tuple[int, int] = (math.ceil(b[0]), math.ceil(b[2]))

    H_shift: npt.NDArray[np.float64] = np.array(
        [
            [1, 0, shifts[0]],
            [0, 1, shifts[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    tform1 = transform.ProjectiveTransform(matrix=H1 @ H_shift)
    tform2 = transform.ProjectiveTransform(matrix=H2 @ H_shift)

    tf_img1: npt.NDArray[np.float64] = transform.warp(
        imgarray1, tform1, output_shape=outshape
    )
    tf_img2: npt.NDArray[np.float64] = transform.warp(
        imgarray2, tform2, output_shape=outshape
    )

    rect_params: RectParams = {
        "H1": H1,
        "H2": H2,
        "H_shift": H_shift,
        "K": K_rescaled,
        "R": R,
    }

    return {
        "image_pairs": [tf_img1, tf_img2],
        "rect_params": rect_params,
    }
