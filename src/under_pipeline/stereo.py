from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from osgeo import gdal  # type: ignore

from under_pipeline.config import UseGeoDataset1Config
from under_pipeline.db_core import (
    check_pair_records,
    get_img_params_db,
    insert_image_pair_to_db,
    get_rect_params,
)
from under_pipeline.io_utils import write_float32_tiff
from under_pipeline.rectify import rectify_stereopair
from under_pipeline.get_3d_UG import write_disparity_PASMNet  # adjust if your function name differs
# If you have an SGM helper, import it:
# from .sgm_disparity import writedisparitySGM


def _write_disparity_cnn(
    rect_img1: np.ndarray,
    rect_img2: np.ndarray,
    tmp_disp_root: Path,
    model_path: Path,
    base_name: str,
) -> np.ndarray:
    """
    Compute disparity using CNN (PASMNet) and write result to disk.

    Returns
    -------
    np.ndarray
        Disparity array (H, W) for rectified left image.
    """
    tmp_disp_root.mkdir(parents=True, exist_ok=True)

    # Output paths follow legacy convention: left_disp directory etc.
    rect_h, rect_w, _ = rect_img1.shape

    # These filenames replicate the original TMP* naming, simplified:
    left_rect_path = tmp_disp_root / f"{base_name}_rect_left.tif"
    right_rect_path = tmp_disp_root / f"{base_name}_rect_right.tif"
    disp_out_dir = tmp_disp_root

    # Save rectified images
    write_float32_tiff(rect_img1.astype(np.float32), left_rect_path)
    write_float32_tiff(rect_img2.astype(np.float32), right_rect_path)

    # Call PASMNet writer (you may need to adapt arguments to your signature)
    write_disparity_PASMNet(
        height=rect_h,
        width=rect_w,
        filenames_txt=None,      # not needed if you pass paths directly
        tmp_root=str(tmp_disp_root),
        model_path=str(model_path),
        left_image_path=str(left_rect_path),
        right_image_path=str(right_rect_path),
    )

    # By convention of original code, disparity is written to a fixed output name
    left_disp_path = disp_out_dir / "left_disparity.tif"
    disp_ds = gdal.Open(str(left_disp_path))
    disp_array = disp_ds.ReadAsArray().astype(np.float32)

    return disp_array


def _write_disparity_sgm(
    rect_img1: np.ndarray,
    rect_img2: np.ndarray,
    tmp_disp_root: Path,
    sgm_config_path: Path,
    base_name: str,
) -> np.ndarray:
    """
    Compute disparity using SGM and write result to disk.

    You will need to adapt this stub to your actual SGM code.
    """
    tmp_disp_root.mkdir(parents=True, exist_ok=True)

    # This is a stub that assumes a helper similar to the CNN one.
    # Replace with your actual SGM pipeline if you still use it.
    raise NotImplementedError("SGM disparity path not yet wired in stereo.py")


def get_save_disp_rect_params(
    left: Dict[str, object],
    right: Dict[str, object],
    config: UseGeoDataset1Config,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Get (or compute and save) left disparity and rectification parameters for an image pair.

    Parameters
    ----------
    left : dict
        {"fname": <filename>, "array": <(H, W, C) ndarray>} for the base image.
    right : dict
        Same structure for the side image.
    config : UseGeoDataset1Config
        Global configuration providing paths and stereo settings.

    Returns
    -------
    disp_array_left : np.ndarray
        Left disparity array in rectified space (H, W).
    rect_params_left : dict
        Rectification parameters dictionary (H1, H2, H_shift, K, R).
    """
    left_name = str(left["fname"])
    right_name = str(right["fname"])

    # Where rectified disparity for left view is stored
    disp_fname_left = f"{left_name[:-4]}{right_name[:-4]}.tif"
    disp_path_left = config.tmp_disp_warp / disp_fname_left

    pair_in_db, pair_in_disk = check_pair_records(
        left_name=left_name,
        right_name=right_name,
        db_name=config.db_name,
        disp_dir=str(config.tmp_disp_warp),
    )

    rect_params_left: Dict[str, np.ndarray] | None = None
    disp_array_left: np.ndarray | None = None

    if pair_in_disk and pair_in_db:
        print(f"image pair {left_name}, {right_name} found both in disk and db")
        disp_img = gdal.Open(str(disp_path_left))
        disp_array_left = disp_img.ReadAsArray().astype(np.float32)

    elif pair_in_db and not pair_in_disk:
        print(f"image pair {left_name}, {right_name} found in db but not in disk")
        # Fetch rect params and recompute only disparity
        rect_params_left = get_rect_params(left_name, right_name, config.db_name)
        left_params = get_img_params_db(left_name, config.db_name)
        right_params = get_img_params_db(right_name, config.db_name)

        print(f"rectifying {left_name} and {right_name} image pair...")
        rectified = rectify_stereopair(
            np.asarray(left["array"]),
            np.asarray(right["array"]),
            left_params,
            right_params,
            config.downsample_factor,
        )
        rect_img1, rect_img2 = rectified["image_pairs"]

        if config.disparity_method == "CNN":
            if config.model_path is None:
                raise ValueError("config.model_path must be set for CNN disparity")
            disp_array_left = _write_disparity_cnn(
                rect_img1=rect_img1,
                rect_img2=rect_img2,
                tmp_disp_root=config.tmp_disp_root,
                model_path=config.model_path,
                base_name=f"{left_name[:-4]}_{right_name[:-4]}",
            )
        else:
            if config.sgm_config_path is None:
                raise ValueError("config.sgm_config_path must be set for SGM disparity")
            disp_array_left = _write_disparity_sgm(
                rect_img1=rect_img1,
                rect_img2=rect_img2,
                tmp_disp_root=config.tmp_disp_root,
                sgm_config_path=config.sgm_config_path,
                base_name=f"{left_name[:-4]}_{right_name[:-4]}",
            )

        write_float32_tiff(
            np.expand_dims(disp_array_left, axis=-1),
            disp_path_left,
        )

    elif pair_in_disk and not pair_in_db:
        print(f"image pair {left_name}, {right_name} found in disk but not in db")
        left_params = get_img_params_db(left_name, config.db_name)
        right_params = get_img_params_db(right_name, config.db_name)

        print(f"rectifying {left_name} and {right_name} image pair...")
        rectified = rectify_stereopair(
            np.asarray(left["array"]),
            np.asarray(right["array"]),
            left_params,
            right_params,
            config.downsample_factor,
        )
        rect_img1, rect_img2 = rectified["image_pairs"]
        rect_params = rectified["rect_params"]

        insert_image_pair_to_db(left_name, right_name, rect_params, config.db_name)

        if config.disparity_method == "CNN":
            if config.model_path is None:
                raise ValueError("config.model_path must be set for CNN disparity")
            disp_array_left = _write_disparity_cnn(
                rect_img1=rect_img1,
                rect_img2=rect_img2,
                tmp_disp_root=config.tmp_disp_root,
                model_path=config.model_path,
                base_name=f"{left_name[:-4]}_{right_name[:-4]}",
            )
        else:
            if config.sgm_config_path is None:
                raise ValueError("config.sgm_config_path must be set for SGM disparity")
            disp_array_left = _write_disparity_sgm(
                rect_img1=rect_img1,
                rect_img2=rect_img2,
                tmp_disp_root=config.tmp_disp_root,
                sgm_config_path=config.sgm_config_path,
                base_name=f"{left_name[:-4]}_{right_name[:-4]}",
            )

        write_float32_tiff(
            np.expand_dims(disp_array_left, axis=-1),
            disp_path_left,
        )

    else:
        print(f"image pair {left_name}, {right_name} not found in disk and db")
        left_params = get_img_params_db(left_name, config.db_name)
        right_params = get_img_params_db(right_name, config.db_name)

        print(f"rectifying {left_name} and {right_name} image pair...")
        rectified = rectify_stereopair(
            np.asarray(left["array"]),
            np.asarray(right["array"]),
            left_params,
            right_params,
            config.downsample_factor,
        )
        rect_img1, rect_img2 = rectified["image_pairs"]
        rect_params_left = rectified["rect_params"]

        insert_image_pair_to_db(left_name, right_name, rect_params_left, config.db_name)

        if config.disparity_method == "CNN":
            if config.model_path is None:
                raise ValueError("config.model_path must be set for CNN disparity")
            disp_array_left = _write_disparity_cnn(
                rect_img1=rect_img1,
                rect_img2=rect_img2,
                tmp_disp_root=config.tmp_disp_root,
                model_path=config.model_path,
                base_name=f"{left_name[:-4]}_{right_name[:-4]}",
            )
        else:
            if config.sgm_config_path is None:
                raise ValueError("config.sgm_config_path must be set for SGM disparity")
            disp_array_left = _write_disparity_sgm(
                rect_img1=rect_img1,
                rect_img2=rect_img2,
                tmp_disp_root=config.tmp_disp_root,
                sgm_config_path=config.sgm_config_path,
                base_name=f"{left_name[:-4]}_{right_name[:-4]}",
            )

        write_float32_tiff(
            np.expand_dims(disp_array_left, axis=-1),
            disp_path_left,
        )

    if disp_array_left is None:
        disp_img = gdal.Open(str(disp_path_left))
        disp_array_left = disp_img.ReadAsArray().astype(np.float32)

    if rect_params_left is None:
        rect_params_left = get_rect_params(left_name, right_name, config.db_name)

    return disp_array_left, rect_params_left
