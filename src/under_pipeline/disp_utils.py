"""
Disparity computation and tiled stitching for a single rectified stereo pair.

Handles the "big image doesn't fit in memory" problem: splits a rectified
pair into overlapping tiles, runs per-tile disparity estimation
(``calc_disparity``, still delegating to the legacy model-invocation code
in ``get_3d_UG.py``), and stitches tile results back into a full-size
disparity map (``stitch_disparity`` + ``fill_most_disparity``, which keeps
tile centers and discards overlap margins to avoid seam artifacts).

Boundary: this module only ever sees one (left, right) pair and knows
nothing about DB caching, multiview aggregation, or file naming
conventions — that's ``disp_cache.py`` and ``multiview_core.py``.
Tiling itself (pure array slicing, no disparity semantics) lives in
``io_utils.tile_image``; this module is what calls it.
"""
from __future__ import annotations

import time

import numpy as np
from osgeo import gdal
from skimage import transform

from under_pipeline.config import UseGeoDataset1Config
from under_pipeline.stereo import _write_disparity_cnn, _write_disparity_sgm


def calc_disparity(
    rect_img1: np.ndarray,
    rect_img2: np.ndarray,
    disparity_method: str,
    config: UseGeoDataset1Config,
    tile_name: str,
) -> np.ndarray:
    """
    Estimate disparity for one rectified tile pair via CNN or SGM.

    Dispatches to the CNN (PASMNet) or SGM (pandora) disparity estimator
    — see ``stereo._write_disparity_cnn`` / ``stereo._write_disparity_sgm``
    — and combines the resulting disparity and validity-mask arrays.

    Parameters
    ----------
    rect_img1, rect_img2 : np.ndarray
        Rectified left/right tile arrays (H, W, C).
    disparity_method : str
        ``"CNN"`` or ``"SGM"``.
    config : UseGeoDataset1Config
        Pipeline configuration; supplies ``tmp_disp_root``, ``model_path``,
        ``sgm_config_path``.
    tile_name : str
        Unique identifier for this tile, used to build output filenames
        so repeated calls (e.g. across tiles in ``stitch_disparity``)
        don't overwrite each other's intermediate files — the legacy
        version reused a single hardcoded filename for every call.
        QUICKFIX: overwritten this in ``stitch_disparity`` to be constant

    Returns
    -------
    np.ndarray
        (H, W, 2) array — band 0 is disparity, band 1 is the
        method-specific validity/mask channel.
    """
    if disparity_method == "CNN":
        if config.model_path is None:
            raise ValueError("config.model_path must be set for CNN disparity")
        disp_array, mask_array = _write_disparity_cnn(
            rect_img1, rect_img2, config.tmp_disp_root, config.model_path, tile_name,
        )
        # upscale mask since it's doesn't match input image resolution
        tmp_mask_fname = config.tmp_disp_root / f"{tile_name}_mask.tif"
        mask_array = gdal.Open(str(tmp_mask_fname))
        mask_array = mask_array.ReadAsArray()
        mask_array = transform.resize(mask_array, disp_array.shape, anti_aliasing=False)
    elif disparity_method == "SGM":
        if config.sgm_config_path is None:
            raise ValueError("config.sgm_config_path must be set for SGM disparity")
        disp_array, mask_array = _write_disparity_sgm(
            rect_img1, rect_img2, config.tmp_disp_root, config.sgm_config_path, tile_name,
        )
    else:
        raise ValueError(f"Unknown disparity_method: {disparity_method!r}")

    return np.dstack((disp_array, mask_array))


def fill_most_disparity(
    disp_array: np.ndarray,
    disp_array_subset: np.ndarray,
    startrow: int,
    endrow: int,
    startcol: int,
    endcol: int,
    sub_height: int,
    sub_width: int,
    sub_overlap_height: int,
    sub_overlap_width: int,
    nrows: int,
    ncols: int,
) -> None:
    """
    Copy one tile's disparity result into the full-size array, discarding
    the tile's overlap margins to avoid seam artifacts at tile boundaries.

    Which margins get trimmed depends on the tile's position (edge tiles
    keep the outer edge; interior tiles trim both sides), hence the nine
    position-dependent branches below. Mutates ``disp_array`` in place.

    Parameters
    ----------
    disp_array : np.ndarray
        Full-size (H, W, 2) output array being assembled.
    disp_array_subset : np.ndarray
        This tile's (sub_height, sub_width, 2) disparity result.
    startrow, endrow, startcol, endcol : int
        This tile's placement in the full-size array.
    sub_height, sub_width : int
        Tile dimensions.
    sub_overlap_height, sub_overlap_width : int
        Overlap between adjacent tiles, in pixels — must match the values
        used to create the tiles (``io_utils.tile_image``).
    nrows, ncols : int
        Full-size array dimensions.
    """
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


def stitch_disparity(
    rect_img1: np.ndarray,
    img1_subsets: np.ndarray,
    img2_subsets: np.ndarray,
    disparity_method: str,
    overlap_height: int,
    overlap_width: int,
    config: UseGeoDataset1Config,
    pair_name: str,
    start_time: float | None = None,
) -> np.ndarray:
    """
    Compute disparity for a rectified pair by running tiles through the
    disparity estimator and stitching the results back together.

    Splits work into overlapping tiles (see ``io_utils.tile_image``) so
    images too large to process at once still fit in memory. Per-tile
    disparity estimation is delegated to ``calc_disparity``; tile results
    are merged with ``fill_most_disparity``, which keeps the center of
    each tile and discards the overlap margins to avoid seam artifacts.

    Parameters
    ----------
    rect_img1 : np.ndarray
        Full rectified left image (H, W, C); only its shape is used.
    img1_subsets, img2_subsets : np.ndarray
        Tiled left/right image arrays as returned by ``tile_image``,
        shape (N, tile_height, tile_width, C).
    disparity_method : str
        ``"CNN"`` or ``"SGM"`` — passed through to ``calc_disparity``.
    overlap_height, overlap_width : int
        Tile overlap, in pixels, used when the tiles were created — must
        match the values passed to ``tile_image`` or the stitched seams
        will be misaligned.
    config : UseGeoDataset1Config
        Pipeline configuration, passed through to ``calc_disparity``.
    pair_name : str
        Identifier for the (left, right) pair being stitched (e.g.
        ``f"{left_fname[:-4]}{right_fname[:-4]}"``), combined with a
        per-tile index to build unique intermediate filenames.
    start_time : float, optional
        Pipeline start time (``time.time()``), used only for elapsed-time
        logging. If omitted, no timing is logged.

    Returns
    -------
    np.ndarray
        (H, W, 2) array — band 0 is disparity, band 1 is the
        method-specific validity/mask channel.
    """
    nrows, ncols, _nbands = rect_img1.shape
    disp_array = np.zeros(shape=(nrows, ncols, 2), dtype=np.float64)
    n_subsets, sub_height, sub_width, _ = img1_subsets.shape

    subset_idx = 0
    startrow = 0
    endrow = sub_height
    while startrow < nrows:
        startcol = 0
        endcol = sub_width
        while startcol < ncols:
            img1_subset = img1_subsets[subset_idx]
            img2_subset = img2_subsets[subset_idx]
            # tile_name = f"{pair_name}_tile{subset_idx}"
            # QUICKFIX: same tile_names while ``calc_disparity`` is not parallelized
            # since the tiled disparity need not persisted and
            # will cost a lot of extra storage otherwise
            # TODO: move this to config.py or
            # uncomment the tile_name above when
            # ``calc_disparity`` needs to be parallelized
            tile_name = "temp_disparity"
            disp_array_subset = calc_disparity(
                img1_subset, img2_subset, disparity_method, config, tile_name,
            )

            if (endrow <= nrows) and (endcol <= ncols):
                fill_most_disparity(
                    disp_array, disp_array_subset,
                    startrow, endrow, startcol, endcol,
                    sub_height, sub_width,
                    overlap_height, overlap_width,
                    nrows, ncols,
                )
            elif (endrow > nrows) and (endcol <= ncols):
                fill_most_disparity(
                    disp_array, disp_array_subset,
                    nrows - sub_height, nrows, startcol, endcol,
                    sub_height, sub_width,
                    overlap_height, overlap_width,
                    nrows, ncols,
                )
            elif (endrow > nrows) and (endcol > ncols):
                fill_most_disparity(
                    disp_array, disp_array_subset,
                    nrows - sub_height, nrows, ncols - sub_width, ncols,
                    sub_height, sub_width,
                    overlap_height, overlap_width,
                    nrows, ncols,
                )
            elif (endrow <= nrows) and (endcol > ncols):
                fill_most_disparity(
                    disp_array, disp_array_subset,
                    startrow, endrow, ncols - sub_width, ncols,
                    sub_height, sub_width,
                    overlap_height, overlap_width,
                    nrows, ncols,
                )

            startcol = endcol - overlap_width
            endcol = startcol + sub_width
            subset_idx += 1

        startrow = endrow - overlap_height
        endrow = startrow + sub_height

        if start_time is not None:
            elapsed_mins = (time.time() - start_time) / 60.0
            print(
                f"Finished processing {subset_idx} of {n_subsets} subsets "
                f"in {elapsed_mins:.2f} mins"
            )

    return disp_array