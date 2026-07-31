"""
Top-level orchestration of the UnDER reconstruction pipeline.

Owns the run order: initialize DB -> load multiview_dicts.list -> filter
to the configured base image(s) -> reconstruct each via
MultiviewReconstructor -> log progress/timing. Contains no geometry,
disparity, or DB-query logic itself — every step delegates to a
lower-level module (db_client, multiview_service). This is the file to
read first to understand pipeline flow; read the others to understand
what each step actually does.
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import UseGeoDataset1Config
from .db_client import UnderDbClient
from .multiview_service import MultiviewReconstructor


class UnderReconstructionPipeline:
    """
    High-level orchestration of the UnDER pipeline for UseGeo Dataset‑1.
    """

    def __init__(
        self,
        config: UseGeoDataset1Config,
        db_client: UnderDbClient,
        reconstructor: MultiviewReconstructor,
    ) -> None:
        self.config = config
        self.db = db_client
        self.reconstructor = reconstructor
        self._start_time: float | None = None

    @property
    def start_time(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        return self._start_time

    def initialize(self) -> None:
        """
        Initialize the database and any other required state.
        """
        self._start_time = time.time()
        self.db.initialize(
            data_root=str(self.config.data_root),
            img_dir=str(self.config.img_dir),
            las_path=str(self.config.las_path),
            downsample_factor=self.config.downsample_factor,
            epsg=self.config.epsg,
            start_time=self._start_time,
        )

    def _load_multiview_dicts(self) -> List[Dict[str, List[str]]]:
        """
        Load the list of multiview dictionaries from the configured path.
        """
        if self.config.multiview_dict_path is None:
            raise ValueError("config.multiview_dict_path must be set.")
        path = Path(self.config.multiview_dict_path)
        with path.open("rb") as f:
            multiview_dicts: List[Dict[str, List[str]]] = pickle.load(f)
        return multiview_dicts

    def _filter_base_image(self, base_image: str) -> bool:
        """
        Optional base-image filter based on config.base_image_filter.
        Supports single image (str) or multiple images (comma-separated str).
        """
        if self.config.base_image_filter is None:
            return True
        # Support comma-separated list of base images
        filter_images = [img.strip() for img in self.config.base_image_filter.split(",")]
        return base_image in filter_images

    def run_from_multiview_dicts(self) -> None:
        """
        Run multiview reconstruction for all entries in multiview_dicts.list,
        applying an optional base-image filter.
        """
        multiview_dicts = self._load_multiview_dicts()
        base_counter = 0

        for multiview_dict in multiview_dicts:
            base_image = next(iter(multiview_dict.keys()))
            if not self._filter_base_image(base_image):
                continue

            print(f"Processing base image {base_image} "
                  f"({base_counter + 1}/{len(multiview_dicts)})")

            points_3d = self.reconstructor.run_multiview(
                multiview_dict=multiview_dict,
                start_time=self.start_time,
            )

            # Optional: do something with points_3d (e.g. stats/logging)
            if isinstance(points_3d, np.ndarray) and points_3d.size > 0:
                print(f"Reconstructed {points_3d.shape[0]} points for {base_image}")

            base_counter += 1

        elapsed_mins = (time.time() - self.start_time) / 60.0
        print(f"Pipeline finished in {elapsed_mins:.2f} mins")
