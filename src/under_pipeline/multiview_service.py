"""
Thin service wrapper around multiview_core.mvs_reconstruct_db.

Exists so the pipeline orchestration layer (core_pipeline.py) depends on
an injectable class (config + db_client in, reconstruct() out) rather
than importing a free function directly — keeps core_pipeline.py
testable/mockable without needing to know multiview_core's internals.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .config import UseGeoDataset1Config
from .db_client import UnderDbClient
from .multiview_core import mvs_reconstruct_db


class MultiviewReconstructor:
    """
    Service responsible for running multiview reconstruction for
    a given base image and its side images.
    """

    def __init__(
        self,
        config: UseGeoDataset1Config,
        db_client: UnderDbClient,
    ) -> None:
        self.config = config
        self.db = db_client

    def run_multiview(
        self,
        multiview_dict: Dict[str, List[str]],
        start_time: float | None = None,
    ) -> np.ndarray:
        """
        Run DB-backed multiview reconstruction for the given mapping.

        Parameters
        ----------
        multiview_dict : dict
            Mapping {base_image: [side_image1, side_image2, ...]}.
        start_time : float, optional
            Pipeline start time used for logging (minutes elapsed).

        Returns
        -------
        np.ndarray
            Array containing point cloud coordinates and RGB.
        """
        return mvs_reconstruct_db(
            multiview_dict=multiview_dict,
            config=self.config,
            db=self.db,
            start_time=start_time,
        )
