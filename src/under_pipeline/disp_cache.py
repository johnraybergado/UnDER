"""
disp_cache.py  —  Disparity result caching / persistence layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
from osgeo import gdal

from under_pipeline.config import UseGeoDataset1Config
from under_pipeline.db_core import (
    check_pair_records,
    get_rect_params,
    insert_image_pair_to_db,
)
from under_pipeline.io_utils import write_float32_tiff
from under_pipeline.get_3d_UG import (
    calc_disp_reduction_db,
    shift_right_image,
    subset_image,
    stitch_disparity,
)
from under_pipeline.rectify import rectify_stereopair


def get_save_disp_rect_params(
    left: dict,
    right: dict,
    dbname: str,
    config: UseGeoDataset1Config,
    db_image_params_fn: Callable[[str], dict],
) -> Tuple[np.ndarray, dict]:
    """Retrieve (or compute and persist) the left disparity map and rect params
    for a stereo pair.

    Handles four cache states:

    1. **Both in DB and on disk** — load array from disk; rect params fetched
       lazily from DB at the end.
    2. **In DB but not on disk** — recompute disparity from DB rect params,
       then write the TIFF.
    3. **On disk but not in DB** — re-rectify the raw images, insert the pair
       (including new rect params) into the DB; the saved TIFF is reused.
    4. **Neither** — full computation from scratch, then persist both.

    Parameters
    ----------
    left:
        Dict with keys ``'fname'`` (str) and ``'array'`` (np.ndarray, H×W×C).
    right:
        Same structure as *left*.
    dbname:
        PostgreSQL database name.
    config:
        Pipeline configuration. ``config.tmp_disp_warp`` (Path) replaces the
        legacy ``TMP_DISP_WARP`` global constant.
    db_image_params_fn:
        Callable ``(imgname: str) -> dict`` — e.g. ``db.get_image_params``.

    Returns
    -------
    disp_array_left : np.ndarray
        Warped left disparity array (H, W).
    rect_params_left : dict
        Rectification parameters (``H1``, ``H2``, ``Hshift``, ``K``, ``R``).
    """
    left_fname: str = left["fname"]
    right_fname: str = right["fname"]
    left_array: np.ndarray = left["array"]
    right_array: np.ndarray = right["array"]

    rect_params_l: Optional[dict] = None
    disp_array_l: Optional[np.ndarray] = None

    disp_fname_l = f"{left_fname[:-4]}{right_fname[:-4]}.tif"
    # config.tmp_disp_warp is a Path — replaces os.path.join(TMP_DISP_WARP, ...)
    disp_path_l: Path = config.tmp_disp_warp / disp_fname_l

    pair_in_db, pair_on_disk = check_pair_records(left_fname,
                                                  right_fname,
                                                  dbname,
                                                  config.tmp_disp_warp)

    if pair_on_disk and pair_in_db:
        # Case 1: everything already cached ─────────────────────────────────
        print(f"Image pair {left_fname}, {right_fname} found both on disk and in DB.")
        disp_img = gdal.Open(str(disp_path_l))
        disp_array_l = disp_img.ReadAsArray()

    elif pair_in_db and not pair_on_disk:
        # Case 2: DB record exists but TIFF was deleted / never written ──────
        print(f"Image pair {left_fname}, {right_fname} found in DB but not on disk.")
        disp_arrays_l, rect_params_l = _compute_left_disparity_db(
            left, right, dbname, config, db_image_params_fn,
        )
        disp_array_l = disp_arrays_l[:, :, 0]
        write_float32_tiff(np.expand_dims(disp_array_l, axis=-1), disp_path_l)

    elif pair_on_disk and not pair_in_db:
        # Case 3: TIFF exists but DB record is missing ───────────────────────
        print(f"Image pair {left_fname}, {right_fname} found on disk but not in DB.")
        left_img_params = db_image_params_fn(left_fname)
        right_img_params = db_image_params_fn(right_fname)
        print(f"Rectifying image pair {left_fname}…")
        img_rect_params = rectify_stereopair(
            left_array, right_array,
            left_img_params, right_img_params,
            config.downsample_factor,
        )
        rect_params = img_rect_params["rect_params"]
        insert_image_pair_to_db(left_fname, right_fname, rect_params, dbname)
        # The already-existing TIFF is loaded by the lazy-load block below.

    else:
        # Case 4: nothing cached — full computation ──────────────────────────
        print(f"Image pair {left_fname}, {right_fname} not found on disk or in DB.")
        disp_arrays_l, rect_params_l = _compute_left_disparity_db(
            left, right, dbname, config, db_image_params_fn,
        )
        insert_image_pair_to_db(left_fname, right_fname, rect_params_l, dbname)
        disp_array_l = disp_arrays_l[:, :, 0]
        write_float32_tiff(np.expand_dims(disp_array_l, axis=-1), disp_path_l)

    # Lazy-load array from disk if not set above (Cases 1 & 3) ───────────────
    if disp_array_l is None:
        disp_img = gdal.Open(str(disp_path_l))
        disp_array_l = disp_img.ReadAsArray()

    # Lazy-load rect params from DB if not returned above (Cases 1 & 3) ──────
    if rect_params_l is None:
        rect_params_l = get_rect_params(left_fname, right_fname, dbname)

    return disp_array_l, rect_params_l


def _rectify_and_reduce_disparity(
    left: dict,
    right: dict,
    dbname: str,
    config: UseGeoDataset1Config,
    db_image_params_fn: Callable[[str], dict],
) -> Tuple[np.ndarray, np.ndarray, dict, int]:
    """Rectify a stereo pair and reduce the right image's disparity offset.

    Wraps ``rectify_stereopair`` plus the legacy disparity-shift reduction
    (``calc_disp_reduction_db`` / ``shift_right_image``, still defined in
    ``get_3d_UG.py``) so callers get rectified, shift-corrected images ready
    for tiling and disparity estimation.

    Returns
    -------
    rect_img1 : np.ndarray
        Rectified left image.
    shifted_rect_img2 : np.ndarray
        Rectified right image, shifted via baseline depth based method.
    rect_params : dict
        Rectification parameters (H1, H2, H_shift, K, R).
    disparity_shift : int
        Pixel shift applied to the right image; must be added back to the
        raw disparity output to recover true disparities.
    """
    left_fname = left["fname"]
    right_fname = right["fname"]

    left_img_params = db_image_params_fn(left_fname)
    right_img_params = db_image_params_fn(right_fname)

    print(f"Rectifying {left_fname} and {right_fname} image pair...")
    img_rect_params_dict = rectify_stereopair(
        left["array"], right["array"],
        left_img_params, right_img_params,
        config.downsample_factor,
    )
    rect_img1, rect_img2 = img_rect_params_dict["image_pairs"]
    rect_params = img_rect_params_dict["rect_params"]

    # Persist rectified images for debugging/inspection, matching legacy behavior.
    rect_img1_path = config.tmp_disp_warp / f"{left_fname[:-4]}{right_fname[:-4]}_rectified.tif"
    write_float32_tiff(rect_img1, rect_img1_path)
    rect_img2_path = config.tmp_disp_warp / f"{right_fname[:-4]}{left_fname[:-4]}_rectified.tif"
    write_float32_tiff(rect_img2, rect_img2_path)

    print("Reducing absolute disparity values...")
    disparity_shift = calc_disp_reduction_db(
        left_img_params, right_img_params, rect_params, dbname,
    )
    shifted_rect_img2 = shift_right_image(rect_img2, -disparity_shift)

    return rect_img1, shifted_rect_img2, rect_params, disparity_shift


def _compute_left_disparity_db(
    left: dict,
    right: dict,
    dbname: str,
    config: UseGeoDataset1Config,
    db_image_params_fn: Callable[[str], dict],
) -> Tuple[np.ndarray, dict]:
    """Compute the left disparity map for a DB-backed stereo pair.

    Replaces the legacy, monolithic ``get_left_disparity_db`` from
    ``get_3d_UG.py``. Rectification and disparity-shift reduction now live
    in ``_rectify_and_reduce_disparity``; tiling (``subset_image``) and
    disparity stitching (``stitch_disparity``) remain legacy imports from
    ``get_3d_UG.py`` pending their own refactor.

    Returns
    -------
    disp_arrays : np.ndarray
        (H, W, 2) array — band 0 is left disparity, band 1 is the
        method-specific validity/mask channel.
    rect_params : dict
        Rectification parameters for the pair.
    """
    rect_img1, shifted_rect_img2, rect_params, disparity_shift = (
        _rectify_and_reduce_disparity(left, right, dbname, config, db_image_params_fn)
    )

    print("Subsetting image pairs...")
    img1_subsets = subset_image(
        rect_img1,
        config.subset_height, config.subset_width,
        config.subset_height_overlap, config.subset_width_overlap,
    )
    img2_subsets = subset_image(
        shifted_rect_img2,
        config.subset_height, config.subset_width,
        config.subset_height_overlap, config.subset_width_overlap,
    )

    print("Calculating left disparity image...")
    disp_arrays = stitch_disparity(rect_img1, img1_subsets, img2_subsets, config.disparity_method)

    # Shift disparity values back to their true (unreduced) scale.
    disp_arrays[:, :, 0] = disp_arrays[:, :, 0] + disparity_shift

    return disp_arrays, rect_params
