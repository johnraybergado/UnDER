"""
Single source of truth for pipeline configuration: paths, DB name, EPSG,
disparity method/model/threshold settings, tiling dimensions, and
multiview run parameters. All derived tmp-directory paths (tmp_ply,
tmp_disp_orig, etc.) are computed once in __post_init__ from tmp_root.

Boundary: no logic beyond path derivation lives here — every other
module takes a UseGeoDataset1Config instance as a parameter rather than
hardcoding paths/constants itself (the pattern the legacy get_3d_UG.py
module-level globals are being migrated away from).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class UseGeoDataset1Config:
    """
    Configuration for running the UnDER pipeline on UseGeo Dataset‑1.
    """

    # Core data
    data_root: Path          # e.g. .../Use_Geo/Dataset-1
    img_dir: Path            # e.g. data_root / "undistorted_images"
    las_path: Path           # e.g. data_root / "DIM_after_adjustment_dataset_C.las"

    # Temporary / working directory
    tmp_root: Path           # e.g. .../03_use_geo/98_tmp_db

    # Coordinate system / DB
    epsg: int = 32632
    db_name: str = "tmp_finetune_ug1"

    # Resolution / scaling
    downsample_factor: int = 1  # image downsampling factor

    # Overlap geometry
    overlap_bounds_buffer: int = 300  # OVERLAP_BOUNDS_BUFFER

    # Disparity / stereo
    disparity_method: str = "CNN"     # "CNN" or "SGM"
    model_path: Path | None = None    # PASMNet model path
    mask_method: str | None = "RLCC"  # "RLCC", "FM", or None
    disp_diff_threshold: float = 0.75  # DISP_DIFF_THRESHOLD
    disparity_shift_ratio: float = 0.9  # DISPARITY_SHIFT_RATIO

    # Tiling / subsets
    subset_height_overlap: int = 512
    subset_width_overlap: int = 1280
    subset_width: int = 2880
    subset_height: int = 1620

    # Multiview setup
    multiview_dict_path: Path | None = None
    base_image_filter: str | None = "2021-04-23_13-17-22_S2223314_DxO.jpg"

    # Derived tmp paths (set in __post_init__)
    tmp_disp_orig: Path | None = None
    tmp_disp_warp: Path | None = None
    tmp_ply: Path | None = None
    tmp_disp_root: Path | None = None
    sgm_config_path: Path | None = None

    def __post_init__(self) -> None:
        if self.tmp_disp_orig is None:
            self.tmp_disp_orig = self.tmp_root / "left_disp" / "orig"
        if self.tmp_disp_warp is None:
            self.tmp_disp_warp = self.tmp_root / "left_disp" / "warp"
        if self.tmp_ply is None:
            self.tmp_ply = self.tmp_root / "ply"
        if self.tmp_disp_root is None:
            self.tmp_disp_root = self.tmp_root / "left_disp"
        if self.sgm_config_path is None:
            self.sgm_config_path = self.tmp_root / "sgmconfig.json"
