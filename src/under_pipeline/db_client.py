from __future__ import annotations

from typing import Any, Dict, List

from .db_core import (
    init_db as core_init_db,
    get_image_filenames as core_get_image_filenames,
    get_img_params_db as core_get_img_params_db,
    get_global_param as core_get_global_param,
)


class UnderDbClient:
    """
    DB access layer for the UnDER pipeline.

    Wraps the psycopg2-based helpers in an object so that the rest of
    the code does not deal with raw database names or connections.
    """

    def __init__(self, db_name: str) -> None:
        self.db_name = db_name

    # ---- lifecycle / setup -------------------------------------------------

    def initialize(
        self,
        data_root: str,
        img_dir: str,
        las_path: str,
        downsample_factor: int,
        epsg: int,
        start_time: float,
    ) -> None:
        """
        Initialize the database (create DB, tables, and metadata) if needed.
        """
        core_init_db(
            self.db_name,
            data_root,
            img_dir,
            las_path,
            start_time,
            downsample_factor,
            epsg,
        )

    # ---- queries -----------------------------------------------------------

    def list_images(self) -> List[str]:
        """
        Return all image filenames registered in the database.
        """
        return core_get_image_filenames(self.db_name)

    def get_image_params(self, img_name: str) -> Dict[str, Any]:
        """
        Return orientation and metadata for a single image.
        """
        return core_get_img_params_db(img_name, self.db_name)

    def get_global_param(self, param_name: str) -> Any:
        """
        Return the value of a global pipeline parameter.
        """
        return core_get_global_param(self.db_name, param_name)

    def delete_point_cloud_for_base_image(self, base_image: str) -> None:
        """
        Delete the point cloud row associated with the given base image.
        """
        core_delete_pc_for_base_image(self.db_name, base_image)
