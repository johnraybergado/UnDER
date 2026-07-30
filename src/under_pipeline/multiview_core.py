"""
Multiview reconstruction for one base image against multiple side images.

For each side image: computes disparity + right-left consistency (RLCC)
mask via ``stereo.py``/``disp_cache.py`` helpers. Then aggregates across
all side images to triangulate a single point cloud for the base image,
writes it to PLY, and manages the multi_stereo/point_cloud DB bookkeeping
(existence checks, inserts, disk/DB consistency reconciliation).

Boundary: this is the "more than one pair at once" layer — anything that
only needs a single (left, right) pair belongs in ``stereo.py`` or
``disp_cache.py`` instead. Called by ``multiview_service.py``, which
just adapts this function to the service/DI style used elsewhere in the
pipeline.

TODO: clean up all db_name parameters.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np
from osgeo import gdal

from under_pipeline.config import UseGeoDataset1Config
from under_pipeline.db_client import UnderDbClient
from under_pipeline.io_utils import (
    load_gdal_rgb,
    load_ply_xyz,
    write_float32_tiff,
    write_ply_fpc_format,
)

# These should point to where you place the corresponding legacy functions
from under_pipeline.get_3d_UG import (
    triangulate_multi_db,
    compare_disparity_maps,
    check_multi_stereo_entry,
    check_point_cloud_entry,
    insert_point_cloud_entry,
    update_pc_status_in_ms,
)
from under_pipeline.stereo import warp_left_disparity
from under_pipeline.db_core import insert_multi_stereo
from under_pipeline.rectify import rectify_stereopair
from under_pipeline.disp_cache import get_save_disp_rect_params


def _compute_right_left_mask(
    base_image: str,
    side_image: str,
    left_array: np.ndarray,
    right_array: np.ndarray,
    config: UseGeoDataset1Config,
    db: UnderDbClient,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Compute left disparity and right-left consistency mask in original left space.

    Returns
    -------
    orig_disp_array_left : (H, W) float array
    orig_mask_array_left : (H, W) float 0/1 array
    rect_params_left : dict
    mappings : dict
    """
    left = {"fname": base_image, "array": left_array}
    right = {"fname": side_image, "array": right_array}

    # Disparities in rectified space
    # OLD (passes raw DF + disparity_method positional args):

    # disp_array_left, rect_params_left = get_save_disp_rect_params(
    #     left,
    #     right,
    #     db.db_name,
    #     config.downsample_factor,
    #     config.disparity_method,
    # )
    # disp_array_right, rect_params_right = get_save_disp_rect_params(
    #     right,
    #     left,
    #     db.db_name,
    #     config.downsample_factor,
    #     config.disparity_method,
    # )

    # NEW (passes config object; db_image_params_fn is explicit keyword arg):
    disp_array_left, rect_params_left = get_save_disp_rect_params(
        left, right, db.db_name, config,
        db_image_params_fn=db.get_image_params,
    )
    disp_array_right, rect_params_right = get_save_disp_rect_params(
        right, left, db.db_name, config,
        db_image_params_fn=db.get_image_params,
    )

    # 1) Warp right disparity back to original right image space
    disp_fname = f"{right['fname'][:-4]}{left['fname'][:-4]}.tif"
    disp_path = config.tmp_disp_orig / disp_fname

    if not disp_path.exists():
        orig_disp_array_right, _ = warp_left_disparity(
            disp_array_right,
            right_array.shape[:-1],
            rect_params_right,
        )
        orig_disp_array_right = np.expand_dims(orig_disp_array_right, axis=-1)
        write_float32_tiff(orig_disp_array_right, disp_path)
    else:
        disp_img = gdal.Open(str(disp_path))
        orig_disp_array_right = disp_img.ReadAsArray()
        orig_disp_array_right = np.expand_dims(orig_disp_array_right, axis=-1)

    # 2) Rectify swapped pair (left vs warped-right) to compare disparities
    disp_fname_wr = f"{right['fname'][:-4]}{left['fname'][:-4]}_warped_RL.tif"
    disp_path_wr = config.tmp_disp_warp / disp_fname_wr

    if not disp_path_wr.exists():
        rect_params_swapped = rectify_stereopair(
            left_array,
            orig_disp_array_right,
            db.get_image_params(left["fname"]),
            db.get_image_params(right["fname"]),
            config.downsample_factor,
        )
        _, disp_array_wr = rect_params_swapped["image_pairs"]
        write_float32_tiff(disp_array_wr, disp_path_wr)
    else:
        disp_img = gdal.Open(str(disp_path_wr))
        disp_array_wr = disp_img.ReadAsArray()

    # 3) Deviations and RLCC mask in rectified space
    deviations_fname = f"{left['fname'][:-4]}{right['fname'][:-4]}_deviations.tif"
    deviations_path = config.tmp_disp_warp / deviations_fname

    if not deviations_path.exists():
        deviations = compare_disparity_maps(disp_array_left, disp_array_wr)
        write_float32_tiff(
            np.expand_dims(deviations, axis=-1),
            deviations_path,
        )
    else:
        deviations_img = gdal.Open(str(deviations_path))
        deviations = deviations_img.ReadAsArray()

    mask_rect = np.zeros_like(deviations, dtype=float)
    mask_rect[np.abs(deviations) < config.disp_diff_threshold] = 1.0

    # 4) Warp disparity and mask back to original left image space
    disp_fname_left = f"{left['fname'][:-4]}{right['fname'][:-4]}.tif"
    disp_path_left = config.tmp_disp_orig / disp_fname_left

    if not disp_path_left.exists():
        orig_disp_array_left, mappings = warp_left_disparity(
            disp_array_left,
            left_array.shape[:-1],
            rect_params_left,
        )
        write_float32_tiff(
            np.expand_dims(orig_disp_array_left, axis=-1),
            disp_path_left,
        )
    else:
        disp_img = gdal.Open(str(disp_path_left))
        orig_disp_array_left = disp_img.ReadAsArray()
        mappings = rect_params_left  # fallback if mappings not persisted

    mask_fname = f"{left['fname'][:-4]}{right['fname'][:-4]}_mask.tif"
    mask_path = config.tmp_disp_orig / mask_fname

    orig_mask_array_left, _ = warp_left_disparity(
        mask_rect,
        left_array.shape[:-1],
        rect_params_left,
    )

    if not mask_path.exists():
        write_float32_tiff(
            np.expand_dims(orig_mask_array_left, axis=-1),
            mask_path,
        )

    return orig_disp_array_left, orig_mask_array_left, rect_params_left, mappings


def mvs_reconstruct_db(
    multiview_dict: Dict[str, List[str]],
    config: UseGeoDataset1Config,
    db: UnderDbClient,
    start_time: float | None = None,
) -> np.ndarray:
    """
    DB-backed multiview reconstruction for a single base image.

    Parameters
    ----------
    multiview_dict : dict
        Mapping {base_image: [side_image1, side_image2, ...]}.
    config : UseGeoDataset1Config
        Pipeline configuration including paths and stereo parameters.
    db : UnderDbClient
        Database client providing image params and global parameters.
    start_time : float, optional
        Pipeline start time (for logging).

    Returns
    -------
    np.ndarray
        Reconstructed 3D points for this base image.
    """
    if start_time is None:
        start_time = time.time()

    base_image = next(iter(multiview_dict.keys()))
    side_images = multiview_dict[base_image]

    disparity_arrays: List[np.ndarray] = []
    mask_arrays: List[np.ndarray] = []
    rect_params_list: List[dict] = []
    mappings_list: List[dict] = []

    # 1. For each side image, compute disparities and RLCC mask
    for side_image in side_images:
        left_path = config.img_dir / base_image
        right_path = config.img_dir / side_image

        left_array = load_gdal_rgb(left_path)
        right_array = load_gdal_rgb(right_path)

        disp_left, mask_left, rect_params_left, mappings = _compute_right_left_mask(
            base_image=base_image,
            side_image=side_image,
            left_array=left_array,
            right_array=right_array,
            config=config,
            db=db,
        )

        elapsed_mins = (time.time() - start_time) / 60.0
        print(
            f"Finished RLCC for pair {base_image} - {side_image} "
            f"in {elapsed_mins:.2f} mins"
        )

        disparity_arrays.append(disp_left)
        mask_arrays.append(mask_left)
        rect_params_list.append(rect_params_left)
        mappings_list.append(mappings)

    # 2. Multi-stereo bookkeeping in DB
    multi_stereo_exists = check_multi_stereo_entry(db.db_name, base_image)
    if not multi_stereo_exists:
        insert_multi_stereo(db.db_name,
                            base_image, side_images,
                            config.overlap_bounds_buffer)
    else:
        print(f"Multi-stereo with base image {base_image} already in DB")

    pc_in_db = check_point_cloud_entry(db.db_name, base_image)
    pc_path = config.tmp_ply / f"{base_image[:-4]}.ply"
    pc_on_disk = pc_path.exists()

    # Case 1: PC in both DB and disk
    if pc_in_db and pc_on_disk:
        print(
            f"Point cloud of base image {base_image} "
            f"found both on disk and in DB"
        )
        return load_ply_xyz(pc_path)

    # Case 2: PC in neither DB nor disk – triangulate and save
    if not pc_in_db and not pc_on_disk:
        print(
            f"Point cloud of base image {base_image} "
            f"not found on disk or in DB"
        )
        print(f"Triangulating multi stereo with base image {base_image}...")

        points_3d, mask = triangulate_multi_db(
            base_image,
            side_images,
            disparity_arrays,
            mask_arrays,
            mappings_list,
            rect_params_list,
            db.db_name,
        )

        elapsed_mins = (time.time() - start_time) / 60.0
        print(f"Triangulation done in {elapsed_mins:.2f} mins")

        print("Writing PLY file...")
        left_array = load_gdal_rgb(config.img_dir / base_image)
        masked_image = left_array[mask == 1]
        points_3d[:, 3:] = masked_image.reshape(-1, 3).astype(np.uint8)

        n_views = len(side_images) + 1
        write_ply_fpc_format(
            vertices=points_3d[:, 0:3],
            colors=points_3d[:, 3:],
            imgname=base_image,
            num_images=n_views,
            get_global_param_fn=db.get_global_param,
            out_dir=config.tmp_ply,
        )

        elapsed_mins = (time.time() - start_time) / 60.0
        print(f"PLY writing done in {elapsed_mins:.2f} mins")

        pc_id = insert_point_cloud_entry(db.db_name, base_image, points_3d[:, 0:3])
        update_pc_status_in_ms(db.db_name, base_image, pc_id)
        return points_3d

    # Case 3: PC on disk but not in DB – read PLY and insert
    if not pc_in_db and pc_on_disk:
        print(
            f"Point cloud of base image {base_image} "
            f"found on disk but not in DB"
        )
        points_3d = load_ply_xyz(pc_path)
        pc_id = insert_point_cloud_entry(db.db_name, base_image, points_3d)
        update_pc_status_in_ms(db.db_name, base_image, pc_id)
        return points_3d

    # Case 4: PC in DB but not on disk – delete DB entry and recompute
    print(
        f"Point cloud of base image {base_image} "
        f"found in DB but not on disk"
    )

    db.delete_point_cloud_for_base_image(base_image)

    print(f"Triangulating multi stereo with base image {base_image}...")
    points_3d, mask = triangulate_multi_db(
        base_image,
        side_images,
        disparity_arrays,
        mask_arrays,
        mappings_list,
        rect_params_list,
        db.db_name,
    )

    elapsed_mins = (time.time() - start_time) / 60.0
    print(f"Triangulation done in {elapsed_mins:.2f} mins")

    print("Writing PLY file...")
    left_array = load_gdal_rgb(config.img_dir / base_image)
    masked_image = left_array[mask == 1]
    points_3d[:, 3:] = masked_image.reshape(-1, 3).astype(np.uint8)

    n_views = len(side_images) + 1
    write_ply_fpc_format(
        vertices=points_3d[:, 0:3],
        colors=points_3d[:, 3:],
        imgname=base_image,
        num_images=n_views,
        get_global_param_fn=db.get_global_param,
        out_dir=config.tmp_ply,
    )

    elapsed_mins = (time.time() - start_time) / 60.0
    print(f"PLY writing done in {elapsed_mins:.2f} mins")

    pc_id = insert_point_cloud_entry(db.db_name, base_image, points_3d[:, 0:3])
    update_pc_status_in_ms(db.db_name, base_image, pc_id)

    return points_3d
