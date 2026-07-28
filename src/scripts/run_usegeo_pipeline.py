from __future__ import annotations

import argparse
from pathlib import Path

from under_pipeline.config import UseGeoDataset1Config
from under_pipeline.db_client import UnderDbClient
from under_pipeline.multiview_service import MultiviewReconstructor
from under_pipeline.core_pipeline import UnderReconstructionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UnDER multiview reconstruction on UseGeo Dataset‑1."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Root directory of UseGeo Dataset‑1.",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        required=True,
        help="Temporary/working directory for DB and intermediates.",
    )
    parser.add_argument(
        "--las-path",
        type=Path,
        required=True,
        help="Path to LAS file (e.g. DIM_after_adjustment_dataset_C.las).",
    )
    parser.add_argument(
        "--multiview-dicts",
        type=Path,
        required=True,
        help="Path to pickled multiview_dicts.list file.",
    )
    parser.add_argument(
        "--db-name",
        type=str,
        default="tmp_finetune_ug1",
        help="PostgreSQL database name to use/create.",
    )
    parser.add_argument(
        "--disparity-method",
        type=str,
        default="CNN",
        choices=["CNN", "SGM"],
        help="Disparity estimation method.",
    )
    parser.add_argument(
        "--disp-diff-threshold",
        type=float,
        default=0.9,
        help="Right-left consistency difference threshold.",
    )
    parser.add_argument(
        "--base-image-filter",
        type=str,
        default=None,
        help="Optional base image filename to process only that image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    img_dir = args.data_root / "undistorted_images"

    config = UseGeoDataset1Config(
        data_root=args.data_root,
        img_dir=img_dir,
        las_path=args.las_path,
        tmp_root=args.tmp_root,
        db_name=args.db_name,
        disparity_method=args.disparity_method,
        disp_diff_threshold=args.disp_diff_threshold,
        multiview_dict_path=args.multiview_dicts,
        base_image_filter=args.base_image_filter,
    )

    db_client = UnderDbClient(db_name=config.db_name)
    reconstructor = MultiviewReconstructor(config=config, db_client=db_client)
    pipeline = UnderReconstructionPipeline(
        config=config,
        db_client=db_client,
        reconstructor=reconstructor,
    )

    pipeline.initialize()
    pipeline.run_from_multiview_dicts()


if __name__ == "__main__":
    main()
