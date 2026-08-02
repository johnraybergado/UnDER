"""
Pairwise stereo geometry for a single (left, right) image pair.

Covers: disparity-shift reduction to recenter values near zero
(``calc_disp_reduction_db``), applying that shift (``shift_right_image``),
warping a rectified disparity map back into original image space
(``warp_left_disparity``), and writing disparity output via the
CNN/SGM-specific helpers (``_write_disparity_cnn``, ``_write_disparity_sgm``).

Boundary: pure two-image-in-two-arrays-out geometry — no knowledge of
"base image with N side images," no PLY writing, no multi_stereo/
point_cloud DB bookkeeping (that's ``multiview_core.py``), and no
per-tile disparity estimation (that's ``disp_utils.py``). Caching
disparity results to disk/DB is handled by ``disp_cache.py``, which
calls into this module rather than duplicating this logic.

TODO: clean up all db_name parameters.
TODO: remove db_core import.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple
import math
import subprocess

from PIL import Image
import numpy as np
from osgeo import gdal  # type: ignore
from skimage import transform
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from models.PASMnet import PASMnet
from under_pipeline.io_utils import write_float32_tiff
from under_pipeline.db_core import get_global_param
# If you have an SGM helper, import it:
# from .sgm_disparity import writedisparitySGM


def _write_disparity_cnn(
    rect_img1: np.ndarray,
    rect_img2: np.ndarray,
    tmp_disp_root: Path,
    model_path: Path,
    base_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute disparity using CNN (PASMNet) and write results to disk.

    Parameters
    ----------
    rect_img1, rect_img2 : np.ndarray
        Rectified left/right tile arrays.
    tmp_disp_root : Path
        Directory for disparity/mask output.
    model_path : Path
        PASMNet checkpoint path.
    base_name : str
        Identifier used to build unique output filenames.

    Returns
    -------
    disp_array : np.ndarray
        Disparity array (H, W) for the rectified left tile.
    mask_array : np.ndarray
        Parallax-attention validity mask (H, W).
    """
    tmp_disp_root.mkdir(parents=True, exist_ok=True)

    disp_array, mask_array = _run_pasmnet_pair(rect_img1, rect_img2, model_path)

    write_float32_tiff(np.expand_dims(disp_array, axis=-1), tmp_disp_root / f"{base_name}_disparity.tif")
    write_float32_tiff(np.expand_dims(mask_array, axis=-1), tmp_disp_root / f"{base_name}_mask.tif")

    return disp_array, mask_array


def _write_disparity_sgm(
    rect_img1: np.ndarray,
    rect_img2: np.ndarray,
    tmp_disp_root: Path,
    sgm_config_path: Path,
    base_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute disparity using SGM (via the ``pandora`` CLI) and write results to disk.

    Parameters
    ----------
    rect_img1, rect_img2 : np.ndarray
        Rectified left/right tile arrays.
    tmp_disp_root : Path
        Directory for intermediate grayscale inputs and pandora output.
    sgm_config_path : Path
        Path to the pandora JSON config file. Note: pandora's config
        specifies its own input/output paths internally, so this config
        file must already point at the grayscale files and output
        directory this function writes to — that wiring isn't managed
        here, same as in the legacy implementation.
    base_name : str
        Identifier used to build unique intermediate filenames.

    Returns
    -------
    disp_array : np.ndarray
        Left disparity array (H, W). Pandora's sign convention is
        flipped relative to ours, so the result is negated.
    mask_array : np.ndarray
        Validity mask (H, W), 1 where valid, 0 where invalid.
    """
    # running sgm on full resolution is horribly slow
    # manually downscale for now
    # works with default config: 
    # subset_width: int = 2880
    # subset_height: int = 1620
    DOWNSCALE_FACTOR = 6
    NAN_FILL = -32

    tmp_disp_root.mkdir(parents=True, exist_ok=True)
    sgm_out_dir = tmp_disp_root / "sgm"
    sgm_out_dir.mkdir(parents=True, exist_ok=True)

    left_gray_path = tmp_disp_root / f"{base_name}_grayscale_left.tif"
    right_gray_path = tmp_disp_root / f"{base_name}_grayscale_right.tif"

    def downsample_mean(x, F):
        H, W, C = x.shape
        x = x[:H - H % F, :W - W % F, :]
        H, W, C = x.shape
        return x.reshape(H // F, F, W // F, F, C).mean(axis=(1, 3))

    rect_img1 = np.round(255 * rect_img1).astype(np.uint8)
    rect_img1 = downsample_mean(rect_img1, DOWNSCALE_FACTOR)
    rect_img1 = np.round(rect_img1).astype(np.uint8)
    rect_img2 = np.round(255 * rect_img2).astype(np.uint8)
    rect_img2 = downsample_mean(rect_img2, DOWNSCALE_FACTOR)
    rect_img2 = np.round(rect_img2).astype(np.uint8)
    Image.fromarray(rect_img1).convert("L").save(left_gray_path)
    Image.fromarray(rect_img2).convert("L").save(right_gray_path)

    command = f"pandora {sgm_config_path} {sgm_out_dir}"
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()
    # print(process.returncode)

    def upsample_repeat(x, F):
        return x.repeat(F, axis=0).repeat(F, axis=1)

    disp_path = sgm_out_dir / "left_disparity.tif"
    mask_path = sgm_out_dir / "left_validity_mask.tif"

    disp_ds = gdal.Open(str(disp_path))
    disp_array = -disp_ds.ReadAsArray()

    mask_ds = gdal.Open(str(mask_path))
    mask_array = mask_ds.ReadAsArray()
    mask_array = np.where(mask_array == 0, 1, 0).astype(mask_array.dtype)

    disp_array = upsample_repeat(disp_array, DOWNSCALE_FACTOR)
    disp_array = disp_array * DOWNSCALE_FACTOR
    disp_array[np.isnan(disp_array)] = NAN_FILL * DOWNSCALE_FACTOR
    mask_array = upsample_repeat(mask_array, DOWNSCALE_FACTOR)

    return disp_array, mask_array


def warp_left_disparity(
    disp_array: np.ndarray,
    output_shape: Tuple[int, int],
    rect_params: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Warp a rectified left disparity map back into the original
    (unrectified) left-image coordinate space.

    Parameters
    ----------
    disp_array : np.ndarray
        Disparity map (H, W) in rectified space.
    output_shape : tuple of int
        Shape (rows, cols) of the original, unrectified left image.
    rect_params : dict
        Rectification parameters as returned by ``rectify_stereopair``
        (keys: ``H1``, ``H2``, ``H_shift``, ``K``, ``R``).

    Returns
    -------
    disp_array_orig : np.ndarray
        Disparity map warped into original left-image space.
    mappings : dict
        ``{"left": left_mapping, "right": right_mapping}`` — homographies
        mapping original image space to rectified space, needed downstream
        for triangulation coordinate transforms.
    """
    H1 = rect_params["H1"]
    H2 = rect_params["H2"]
    H_shift = rect_params["H_shift"]

    left_mapping = H1 @ H_shift
    right_mapping = H2 @ H_shift

    tform = transform.ProjectiveTransform(matrix=np.linalg.inv(left_mapping))
    disp_array_orig = transform.warp(disp_array, tform, output_shape=output_shape)

    return disp_array_orig, {"left": left_mapping, "right": right_mapping}


def calc_disp_reduction_db(
    left_img_params: dict,
    right_img_params: dict,
    rect_params: Dict[str, np.ndarray],
    dbname: str,
    disparity_shift_ratio: float,
) -> int:
    """
    Estimate a constant pixel shift to reduce disparities based on baseline
    depth method.

    Projects the left image's principal point into the right image, using
    the mean flying height (fetched from the DB's ``globals`` table) as an
    assumed scene depth, then measures how far apart the two rectified
    principal points land. Subtracting this shift from raw disparities
    keeps disparity magnitudes small, which the disparity-estimation model
    handles better.

    Parameters
    ----------
    left_img_params, right_img_params : dict
        Camera parameter dictionaries as returned by
        ``UnderDbClient.get_image_params`` (keys: ``X``, ``Y``, ``Z``,
        ``camera_matrix``, ``rotation_matrix``, ...).
    rect_params : dict
        Rectification parameters as returned by ``rectify_stereopair`` in
        ``rectify.py``.
    dbname : str
        Database name, used to fetch ``mean_flying_height`` from globals.
    disparity_shift_ratio : float
        Fraction of the computed shift to actually apply (was previously
        a hardcoded module-level constant; now config-driven — see
        ``config.disparity_shift_ratio``).

    Returns
    -------
    int
        Disparity shift, in pixels, to subtract from the right image
        before disparity estimation.
    """
    z_f = get_global_param(dbname, "mean_flying_height")
    K_left = left_img_params["camera_matrix"]
    R_left = left_img_params["rotation_matrix"]
    C_left = np.array(
        [[left_img_params["X"]], [left_img_params["Y"]], [left_img_params["Z"]]],
        dtype=np.float64,
    )
    P_right = calc_P_matrix(right_img_params)
    H1 = rect_params["H1"]
    H2 = rect_params["H2"]
    H_shift = rect_params["H_shift"]

    # Use the principal point of the left image as the reference pixel.
    x_img_left = 0
    y_img_left = 0
    img_coords_left = np.array([[K_left[0, 2]], [K_left[1, 2]], [1]], dtype=np.float64)

    # Rectified-space coordinates of the left principal point.
    warped_coords_left = np.linalg.inv(H1 @ H_shift) @ img_coords_left
    warped_coords_left = warped_coords_left / warped_coords_left[-1]

    # Back-project to world space, assuming depth equal to mean flying height.
    f_x = K_left[0][0]
    f_y = K_left[1][1]
    coords_delta = np.array(
        [[(x_img_left * z_f / f_x)], [(y_img_left * z_f / f_y)], [z_f]],
        dtype=np.float64,
    )
    coords_world = np.linalg.inv(R_left) @ coords_delta + C_left
    coords_world_h = np.ones(
        shape=(coords_world.shape[0] + 1, coords_world.shape[1]), dtype=np.float64
    )
    coords_world_h[:-1, :] = coords_world

    # Project into the right image, then into its rectified space.
    img_coords_right = P_right @ coords_world_h
    img_coords_right = img_coords_right / img_coords_right[-1, :]
    warped_coords_right = np.linalg.inv(H2 @ H_shift) @ img_coords_right
    warped_coords_right = warped_coords_right / warped_coords_right[-1, :]

    disp_shift = warped_coords_left[0, 0] - warped_coords_right[0, 0]
    return int(disp_shift * disparity_shift_ratio)


def shift_right_image(rect_img: np.ndarray, disparity_shift: float) -> np.ndarray:
    """
    Apply a constant horizontal pixel shift to a rectified image.

    Used to horizontally move the right image before disparity
    estimation, using the shift computed by ``calc_disp_reduction_db``.

    Parameters
    ----------
    rect_img : np.ndarray
        Rectified image array (H, W, C).
    disparity_shift : float
        Horizontal shift, in pixels, to apply.

    Returns
    -------
    np.ndarray
        Shifted image, same shape as ``rect_img``.
    """
    shift_matrix = np.array(
        [
            [1, 0, disparity_shift],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    tform = transform.ProjectiveTransform(matrix=shift_matrix)
    return transform.warp(rect_img, tform, output_shape=rect_img.shape)


# ImageNet mean/std — matches datasets.data_io.get_transform(), which
# TMPDataset applies to every image before feeding it to PASMNet.
_PASMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_PASMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _run_pasmnet_pair(
    rect_img1: np.ndarray,
    rect_img2: np.ndarray,
    model_path: Path,
    device: str = "cuda:0",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run PASMNet disparity estimation on a single rectified tile pair.

    Replaces the legacy ``write_disparity_PASMNet``, which was built
    around ``TMPDataset`` — a file-list dataset meant for batches of
    pairs read from paths recorded in a filenames.txt. Since
    ``calc_disparity`` now calls the CNN estimator once per in-memory
    tile pair, that batch/file-list machinery is unneeded here; this
    runs the model directly on the two arrays, replicating
    ``TMPDataset.__getitem__``'s non-training branch:
    ImageNet-normalize, then edge-pad up to a multiple of 128 in each
    dimension (PASMNet requires input dimensions divisible by 128).

    NOTE: loads and discards the model checkpoint on every call. Fine
    for getting things running, but if this is a bottleneck once tiling
    is in the loop, the model should be loaded once per
    ``stitch_disparity`` call (or higher) and passed in rather than
    reloaded per tile.

    Parameters
    ----------
    rect_img1, rect_img2 : np.ndarray
        Rectified left/right tile arrays (H, W, C), float in [0, 1].
    model_path : Path
        Path to a PASMNet ``.tar`` checkpoint.
    device : str
        Torch device string.

    Returns
    -------
    disp_array : np.ndarray
        Disparity map (H, W), cropped back to the input tile's size.
    mask_array : np.ndarray
        Parallax-attention validity mask (H, W), same size.
    """
    net = PASMnet().to(device)
    if model_path.endswith("tar"):
        ckpt = torch.load(model_path)["state_dict"]
    else:
        ckpt = torch.load(model_path)
    net.load_state_dict(ckpt)
    net.eval()
    cudnn.benchmark = True

    h, w = rect_img1.shape[:2]
    height_snap = h if h % 128 == 0 else 128 * math.ceil(h / 128)
    width_snap = w if w % 128 == 0 else 128 * math.ceil(w / 128)
    top_pad = height_snap - h
    right_pad = width_snap - w

    def to_normalized_tensor(img: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        t = (t - _PASMNET_MEAN) / _PASMNET_STD
        # 'edge' padding in TMPDataset (np.lib.pad mode='edge') == replicate here.
        if top_pad > 0 or right_pad > 0:
            t = F.pad(t, (0, right_pad, top_pad, 0), mode="replicate")
        return t.to(device)

    img_left = to_normalized_tensor(rect_img1)
    img_right = to_normalized_tensor(rect_img2)

    with torch.no_grad():
        disp, mask = net(img_left, img_right, max_disp=0)
        if top_pad > 0:
            disp = disp[:, :, top_pad:, :]
            mask = mask[:, :, top_pad:, :]
        if right_pad > 0:
            disp = disp[:, :, :, :-right_pad]
            mask = mask[:, :, :, :-right_pad]
        disp_array = torch.clamp(disp.squeeze(), 0).cpu().numpy()
        mask_array = torch.clamp(mask.squeeze(), 0).cpu().numpy()

    return disp_array, mask_array


def calc_P_matrix(img_params: dict) -> np.ndarray:
    """
    Build a camera's 3x4 projection matrix P = K [R | t], where t = -R @ C.

    Parameters
    ----------
    img_params : dict
        Camera parameter dictionary with keys ``camera_matrix``,
        ``rotation_matrix``, ``X``, ``Y``, ``Z`` (as returned by
        ``UnderDbClient.get_image_params``).

    Returns
    -------
    np.ndarray
        (3, 4) projection matrix mapping homogeneous world coordinates
        to homogeneous image coordinates.
    """
    K = img_params["camera_matrix"]
    R = img_params["rotation_matrix"]
    C = np.array(
        [[img_params["X"]], [img_params["Y"]], [img_params["Z"]]],
        dtype=np.float64,
    )
    t = -R @ C
    R_t = np.zeros((3, 4), dtype=np.float64)
    R_t[:, :-1] = R
    R_t[:, -1] = t[:, 0]
    return K @ R_t


def compare_disparity_maps(lr_disp_array: np.ndarray, rl_disp_array: np.ndarray) -> np.ndarray:
    """
    Compute right-left consistency deviations between two disparity maps.

    For each pixel in the left-to-right (LR) disparity map, looks up the
    corresponding pixel in the right-to-left (RL) disparity map at the
    LR-implied shifted location, and returns the difference. Small
    deviations indicate the two independently estimated disparities
    agree; large deviations flag likely occlusion/error and are used
    downstream (``multiview_core._compute_right_left_mask``) to build
    the RLCC validity mask.

    Parameters
    ----------
    lr_disp_array : np.ndarray
        (H, W) disparity map estimated left-to-right (base image left).
    rl_disp_array : np.ndarray
        (H, W) disparity map estimated right-to-left (base image right),
        same shape as ``lr_disp_array``.

    Returns
    -------
    np.ndarray
        (H, W) array of per-pixel deviations between the LR disparity
        and the RL disparity sampled at the LR-shifted location.

    Note
    ----
    Ported as-is from the legacy implementation, including its
    pixel-by-pixel Python loop and nearest-neighbor (``int()``-truncated)
    sampling of ``rl_disp_array``. This is O(H*W) in pure Python and will
    be slow on full-size tiles — a candidate for vectorizing with
    ``np.take``/fancy indexing later, but left untouched here to avoid
    changing numerical behavior in the same edit that fixes the import
    wiring.
    """
    ny, nx = lr_disp_array.shape
    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    xv, yv = np.meshgrid(x, y)
    disp_values = lr_disp_array.flatten()

    left_disparities = np.stack((xv.flatten(), yv.flatten(), disp_values))
    right_disparities = left_disparities.copy()
    right_disparities[0:1, :] = right_disparities[0:1, :] - right_disparities[-1:, :]
    for i in range(right_disparities.shape[1]):
        right_disparities[-1:, i] = rl_disp_array[
            int(right_disparities[1, i]), int(right_disparities[0, i])
        ]

    deviations = left_disparities[-1, :] - right_disparities[-1, :]
    deviations = np.reshape(deviations, lr_disp_array.shape)

    return deviations
