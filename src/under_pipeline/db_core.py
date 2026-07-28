from __future__ import annotations

import os
from typing import Any, Dict, List, Sequence

import numpy as np
import psycopg2
import psycopg2.extras
from psycopg2 import Error
from shapely.geometry import Polygon

from .camera_params_UG import init_img_params, estimate_GSD


# ---------------------------------------------------------------------------
# Schema and metadata helpers
# ---------------------------------------------------------------------------


def create_db_tables(db_name: str) -> None:
    """
    Create the tables required by the UnDER pipeline.

    Parameters
    ----------
    db_name : str
        Name of the database where tables should be created.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query_create_tables = """
        CREATE TABLE globals (
            estimated_GSD REAL,
            mean_flying_height REAL,
            EPSG INTEGER,
            X_shift DOUBLE PRECISION,
            Y_shift DOUBLE PRECISION
        );

        CREATE TABLE image (
            img_id SERIAL PRIMARY KEY,
            img_name TEXT,
            img_param_id INTEGER
        );

        CREATE TABLE image_param (
            img_param_id SERIAL PRIMARY KEY,
            nrows INTEGER,
            ncols INTEGER,
            X DOUBLE PRECISION,
            Y DOUBLE PRECISION,
            Z DOUBLE PRECISION,
            camera_matrix DOUBLE PRECISION[3][3],
            rotation_matrix DOUBLE PRECISION[3][3],
            geobound DOUBLE PRECISION[4][3]
        );

        CREATE TABLE image_pair (
            img_pair_id SERIAL PRIMARY KEY,
            base_img_id INTEGER NOT NULL,
            side_img_id INTEGER NOT NULL,
            overlap REAL,
            rect_param_id INTEGER,
            UNIQUE (base_img_id, side_img_id)
        );

        CREATE TABLE rectification_param (
            rect_param_id SERIAL PRIMARY KEY,
            H1 DOUBLE PRECISION[3][3],
            H2 DOUBLE PRECISION[3][3],
            H_shift DOUBLE PRECISION[3][3],
            K DOUBLE PRECISION[3][3],
            R DOUBLE PRECISION[3][3]
        );

        CREATE TABLE multi_stereo (
            ms_id SERIAL PRIMARY KEY,
            base_img_id INTEGER NOT NULL,
            overlapping_bounds INTEGER[],
            pc_ready BOOLEAN,
            pc_id INTEGER
        );

        CREATE TABLE point_cloud (
            pc_id SERIAL PRIMARY KEY,
            pc_name TEXT,
            footprint DOUBLE PRECISION[4][2],
            n_points INTEGER
        );
    """
    cur.execute(query_create_tables)
    print("DB tables created")

    cur.close()
    conn.close()


def populate_imgparams_table(
    img_params: Dict[str, Dict[str, Any]],
    db_name: str,
) -> None:
    """
    Populate the image parametrization table with orientation and metadata.

    Parameters
    ----------
    img_params : dict
        Mapping from image name to its parameter dictionary as returned by
        init_img_params.
    db_name : str
        Target database name.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query_insert_img_params = """
        INSERT INTO image_param
            (nrows, ncols,
             X, Y, Z,
             camera_matrix, rotation_matrix,
             geobound)
        VALUES
            (%(nrows)s, %(ncols)s,
             %(X)s, %(Y)s, %(Z)s,
             %(camera_matrix)s, %(rotation_matrix)s,
             %(geobound)s)
        RETURNING img_param_id;
    """

    query_insert_image = """
        INSERT INTO image (img_name, img_param_id)
        VALUES (%(img_name)s, %(img_param_id)s);
    """

    for img_name, params in img_params.items():
        nrows, ncols = params["nrows"], params["ncols"]
        x, y, z = params["X"], params["Y"], params["Z"]

        camera_matrix = str(params["camera_matrix"].tolist()).replace("[", "{").replace("]", "}")
        rotation_matrix = str(params["rotation_matrix"].tolist()).replace("[", "{").replace("]", "}")

        geobound_array = np.array(params["geobounds"]).reshape((4, 3))
        geobound = str([row.tolist() for row in geobound_array]).replace("[", "{").replace("]", "}")

        cur.execute(
            query_insert_img_params,
            {
                "nrows": nrows,
                "ncols": ncols,
                "X": x,
                "Y": y,
                "Z": z,
                "camera_matrix": camera_matrix,
                "rotation_matrix": rotation_matrix,
                "geobound": geobound,
            },
        )
        img_param_id = cur.fetchone()[0]

        cur.execute(
            query_insert_image,
            {"img_name": img_name, "img_param_id": img_param_id},
        )

    print("image and image_param tables populated")

    cur.close()
    conn.close()


def populate_globals_table(
    db_name: str,
    estimated_gsd: float,
    mean_flying_height: float,
    epsg: int,
    x_shift: float,
    y_shift: float,
) -> None:
    """
    Populate the globals table with pipeline-wide parameters.

    Parameters
    ----------
    db_name : str
        Target database name.
    estimated_gsd : float
        Estimated ground sampling distance.
    mean_flying_height : float
        Mean flying height used for basis depth calculation.
    epsg : int
        EPSG code of the output coordinate reference system.
    x_shift : float
        Global X offset applied to point coordinates.
    y_shift : float
        Global Y offset applied to point coordinates.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query_insert_globals = """
        INSERT INTO globals
            (estimated_GSD,
             mean_flying_height,
             EPSG,
             X_shift,
             Y_shift)
        VALUES
            (%(estimated_GSD)s,
             %(mean_flying_height)s,
             %(epsg)s,
             %(X_shift)s,
             %(Y_shift)s);
    """
    cur.execute(
        query_insert_globals,
        {
            "estimated_GSD": estimated_gsd,
            "mean_flying_height": mean_flying_height,
            "epsg": epsg,
            "X_shift": x_shift,
            "Y_shift": y_shift,
        },
    )

    cur.close()
    conn.close()


# ---------------------------------------------------------------------------
# High-level DB initialization and simple queries
# ---------------------------------------------------------------------------


def init_db(
    db_name: str,
    data_root: str,
    img_dir: str,
    las_path: str,
    start_time: float,
    downsample_factor: int,
    epsg: int,
) -> None:
    """
    Initialize the PostgreSQL/PostGIS database for the UnDER pipeline.

    This creates the DB if missing, sets up tables, and populates
    image parameters and global metadata.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database="postgres",
    )
    conn.autocommit = True
    cur = conn.cursor()

    query_check_db_exists = """
        SELECT EXISTS(
            SELECT datname
            FROM pg_catalog.pg_database
            WHERE lower(datname) = lower(%s)
        );
    """
    cur.execute(query_check_db_exists, (db_name,))
    db_exists = cur.fetchone()[0]

    if db_exists:
        print("Point cloud DB already initialized")
        cur.close()
        conn.close()
        return

    query_create_db = f"CREATE DATABASE {db_name};"
    query_grant_priv = f"GRANT ALL PRIVILEGES ON DATABASE {db_name} TO postgres;"

    cur.execute(query_create_db)
    cur.execute(query_grant_priv)
    cur.close()
    conn.close()

    create_db_tables(db_name)

    metric_calib = os.path.join(data_root, "image_orientations.xyz")
    img_params = init_img_params(
        metric_calib, img_dir, las_path, downsample_factor, start_time
    )
    estimated_gsd = estimate_GSD(img_params)

    populate_imgparams_table(img_params, db_name)

    first_key = next(iter(img_params.keys()))
    mean_flying_height = img_params[first_key]["mean_flying_height"]

    img_xs = [params["X"] for params in img_params.values()]
    img_ys = [params["Y"] for params in img_params.values()]

    x_shift = float(np.min(img_xs))
    y_shift = float(np.min(img_ys))

    populate_globals_table(
        db_name,
        estimated_gsd,
        mean_flying_height,
        epsg,
        x_shift,
        y_shift,
    )
    print("Point cloud DB initialized")


def get_image_filenames(db_name: str) -> List[str]:
    """
    Return all image filenames registered in the database.

    Parameters
    ----------
    db_name : str
        Name of the database to query.

    Returns
    -------
    list of str
        Image filenames stored in the image table.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query = "SELECT img_name FROM image;"
    cur.execute(query)
    image_filenames = [row[0] for row in cur.fetchall()]

    cur.close()
    conn.close()
    return image_filenames


def get_img_params_db(img_name: str, db_name: str) -> Dict[str, Any]:
    """
    Fetch image orientation and metadata for a single image from the database.
    """
    conn = None
    cur = None
    img_params: Dict[str, Any] = {}

    try:
        conn = psycopg2.connect(
            user="postgres",
            password="postgres",
            host="127.0.0.1",
            port="5432",
            database=db_name,
        )
        conn.autocommit = True
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

        query = """
            SELECT *
            FROM image_param
            WHERE img_param_id = (
                SELECT img_param_id
                FROM image
                WHERE img_name = %s
            );
        """
        cur.execute(query, (img_name,))
        res = cur.fetchall()[0]

        img_params = {
            "nrows": res["nrows"],
            "ncols": res["ncols"],
            "X": res["x"],
            "Y": res["y"],
            "Z": res["z"],
            "camera_matrix": np.array(res["camera_matrix"]),
            "rotation_matrix": np.array(res["rotation_matrix"]),
            "geobounds": np.array(res["geobound"]),
        }

    except (Exception, Error) as error:
        print("Error while connecting to DB:", error)
    finally:
        if cur is not None:
            cur.close()
        if conn is not None:
            conn.close()

    return img_params


def get_global_param(db_name: str, param_name: str) -> Any:
    """
    Fetch a global pipeline parameter stored in the database.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query = f"SELECT {param_name} FROM globals LIMIT 1;"
    cur.execute(query)
    value = cur.fetchone()[0]

    cur.close()
    conn.close()
    return value


# ---------------------------------------------------------------------------
# Multiview / image-pair helpers
# ---------------------------------------------------------------------------


def check_pair_records(
    left_name: str,
    right_name: str,
    db_name: str,
    disp_dir: str,
) -> tuple[bool, bool]:
    """
    Check if an image pair exists in DB and/or on disk (disparity file).

    Returns
    -------
    (pair_in_db, pair_in_disk)
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query_get_base_id = "SELECT img_id FROM image WHERE img_name = %s;"
    query_get_side_id = "SELECT img_id FROM image WHERE img_name = %s;"

    cur.execute(query_get_base_id, (left_name,))
    base_img_id = cur.fetchone()[0]

    cur.execute(query_get_side_id, (right_name,))
    side_img_id = cur.fetchone()[0]

    query_check_pair = """
        SELECT EXISTS(
            SELECT 1
            FROM image_pair
            WHERE base_img_id = %s AND side_img_id = %s
        );
    """
    cur.execute(query_check_pair, (base_img_id, side_img_id))
    pair_in_db = cur.fetchone()[0]

    cur.close()
    conn.close()

    disp_fname = f"{left_name[:-4]}{right_name[:-4]}.tif"
    disp_path = os.path.join(disp_dir, disp_fname)
    pair_in_disk = os.path.exists(disp_path)

    return pair_in_db, pair_in_disk


def insert_image_pair_to_db(
    left_name: str,
    right_name: str,
    rect_params: Dict[str, np.ndarray],
    db_name: str,
) -> None:
    """
    Insert rectification parameters and an image_pair row into the DB.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Save rectification parameters
    query_save_rect_params = """
        INSERT INTO rectification_param
            (H1, H2, H_shift, K, R)
        VALUES
            (%(H1)s, %(H2)s, %(H_shift)s, %(K)s, %(R)s)
        RETURNING rect_param_id;
    """

    h1 = str(rect_params["H1"].tolist()).replace("[", "{").replace("]", "}")
    h2 = str(rect_params["H2"].tolist()).replace("[", "{").replace("]", "}")
    h_shift = str(rect_params["H_shift"].tolist()).replace("[", "{").replace("]", "}")
    k = str(rect_params["K"].tolist()).replace("[", "{").replace("]", "}")
    r = str(rect_params["R"].tolist()).replace("[", "{").replace("]", "}")

    cur.execute(
        query_save_rect_params,
        {"H1": h1, "H2": h2, "H_shift": h_shift, "K": k, "R": r},
    )
    rect_param_id = cur.fetchone()[0]

    # Get image ids
    query_get_img_id = "SELECT img_id FROM image WHERE img_name = %s;"

    cur.execute(query_get_img_id, (left_name,))
    base_img_id = cur.fetchone()[0]

    cur.execute(query_get_img_id, (right_name,))
    side_img_id = cur.fetchone()[0]

    # Compute overlap ratio
    query_get_geobound = """
        SELECT ip.geobound
        FROM image_param ip
        LEFT JOIN image i ON ip.img_param_id = i.img_param_id
        WHERE i.img_name = %s;
    """

    cur.execute(query_get_geobound, (left_name,))
    left_extent = np.array(cur.fetchone()[0])

    cur.execute(query_get_geobound, (right_name,))
    right_extent = np.array(cur.fetchone()[0])

    left_poly = Polygon([tuple(row[:-1]) for row in left_extent])
    right_poly = Polygon([tuple(row[:-1]) for row in right_extent])
    overlap = left_poly.intersection(right_poly)
    overlap_ratio = overlap.area / left_poly.area if left_poly.area > 0 else 0.0

    query_save_pair = """
        INSERT INTO image_pair
            (base_img_id, side_img_id, overlap, rect_param_id)
        VALUES
            (%(base_img_id)s, %(side_img_id)s, %(overlap)s, %(rect_param_id)s);
    """
    cur.execute(
        query_save_pair,
        {
            "base_img_id": base_img_id,
            "side_img_id": side_img_id,
            "overlap": overlap_ratio,
            "rect_param_id": rect_param_id,
        },
    )

    cur.close()
    conn.close()


def get_image_pair_id(
    left_name: str,
    right_name: str,
    db_name: str,
) -> int:
    """
    Return img_pair_id for a given (base, side) image name pair.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    query = """
        SELECT img_pair_id
        FROM image_pair ip
        LEFT JOIN image li ON ip.base_img_id = li.img_id
        LEFT JOIN image ri ON ip.side_img_id = ri.img_id
        WHERE li.img_name = %s AND ri.img_name = %s;
    """
    cur.execute(query, (left_name, right_name))
    row = cur.fetchone()

    cur.close()
    conn.close()
    return row["img_pair_id"]


def get_rect_params(
    left_name: str,
    right_name: str,
    db_name: str,
) -> Dict[str, np.ndarray]:
    """
    Fetch rectification parameters for a given image pair.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    img_pair_id = get_image_pair_id(left_name, right_name, db_name)

    query = """
        SELECT *
        FROM rectification_param rp
        LEFT JOIN image_pair ip
            ON rp.rect_param_id = ip.rect_param_id
        WHERE ip.img_pair_id = %s;
    """
    cur.execute(query, (img_pair_id,))
    res = cur.fetchone()

    rect_params = {
        "H1": np.array(res["h1"]),
        "H2": np.array(res["h2"]),
        "H_shift": np.array(res["h_shift"]),
        "K": np.array(res["k"]),
        "R": np.array(res["r"]),
    }

    cur.close()
    conn.close()
    return rect_params


# ---------------------------------------------------------------------------
# Multiview / point-cloud bookkeeping
# ---------------------------------------------------------------------------


def check_point_cloud_entry(db_name: str, base_image: str) -> bool:
    """
    Check if a point cloud entry exists in DB for a given base image.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query = """
        SELECT
            EXISTS(
                SELECT 1
                FROM point_cloud pc
                LEFT JOIN multi_stereo ms ON ms.pc_id = pc.pc_id
                LEFT JOIN image im ON ms.base_img_id = im.img_id
                WHERE im.img_name = %s
            );
    """
    cur.execute(query, (base_image,))
    exists = cur.fetchone()[0]

    cur.close()
    conn.close()
    return bool(exists)


def insert_point_cloud_entry(
    db_name: str,
    base_image: str,
    points_xyz: np.ndarray,
) -> int:
    """
    Insert a point_cloud entry and return its id.

    Parameters
    ----------
    points_xyz : (N, 3) array
        XYZ coordinates of points.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    pc_name = f"{base_image[:-4]}.ply"
    xmin = float(points_xyz[:, 0].min())
    xmax = float(points_xyz[:, 0].max())
    ymin = float(points_xyz[:, 1].min())
    ymax = float(points_xyz[:, 1].max())
    footprint = [
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
    ]
    footprint_str = str(footprint).replace("[", "{").replace("]", "}")
    n_points = int(points_xyz.shape[0])

    query_insert_pc = """
        INSERT INTO point_cloud (pc_name, footprint, n_points)
        VALUES (%(pc_name)s, %(footprint)s, %(n_points)s)
        RETURNING pc_id;
    """
    cur.execute(
        query_insert_pc,
        {
            "pc_name": pc_name,
            "footprint": footprint_str,
            "n_points": n_points,
        },
    )
    pc_id = cur.fetchone()[0]

    cur.close()
    conn.close()
    return pc_id


def update_pc_status_in_ms(
    db_name: str,
    base_image: str,
    pc_id: int,
) -> None:
    """
    Mark a point cloud as ready in multi_stereo for the given base image.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query = """
        UPDATE multi_stereo
        SET pc_id = %s, pc_ready = TRUE
        WHERE base_img_id = (
            SELECT img_id FROM image WHERE img_name = %s
        );
    """
    cur.execute(query, (pc_id, base_image))

    cur.close()
    conn.close()


def insert_multi_stereo(
    db_name: str,
    base_image: str,
    side_images: Sequence[str],
    overlap_bounds_buffer_m: float,
) -> None:
    """
    Insert a multi_stereo record for a base image, computing overlapping bounds.

    Parameters
    ----------
    overlap_bounds_buffer_m : float
        Buffer in meters to shrink the overlapping polygon before
        selecting overlapping image ids.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    base_params = get_img_params_db(base_image, db_name)
    base_extent = Polygon([tuple(row[:-1]) for row in base_params["geobounds"]])

    side_extents = []
    for side_image in side_images:
        side_params = get_img_params_db(side_image, db_name)
        side_poly = Polygon([tuple(row[:-1]) for row in side_params["geobounds"]])
        side_extents.append(side_poly)

    if len(side_extents) > 1:
        side_overlap = side_extents[0]
        for poly in side_extents[1:]:
            side_overlap = side_overlap.intersection(poly)
    else:
        side_overlap = side_extents[0]

    final_bounds = base_extent.intersection(side_overlap)

    # Use GSD from globals to convert buffer in meters to map units
    query_get_gsd = "SELECT estimated_GSD FROM globals LIMIT 1;"
    cur.execute(query_get_gsd)
    gsd = float(cur.fetchone()[0])

    final_bounds = final_bounds.buffer(-overlap_bounds_buffer_m * gsd)

    # Collect overlapping image ids
    query_all_images = "SELECT img_id, img_name FROM image;"
    cur.execute(query_all_images)

    overlapping_ids: list[int] = []
    for img_id, img_name in cur.fetchall():
        params = get_img_params_db(img_name, db_name)
        img_extent = Polygon([tuple(row[:-1]) for row in params["geobounds"]])
        if final_bounds.intersects(img_extent):
            overlapping_ids.append(img_id)

    query_get_base_id = "SELECT img_id FROM image WHERE img_name = %s;"
    cur.execute(query_get_base_id, (base_image,))
    base_img_id = cur.fetchone()[0]

    print("Inserting multi-stereo record to DB...")
    query_insert_ms = """
        INSERT INTO multi_stereo (base_img_id, overlapping_bounds, pc_ready, pc_id)
        VALUES (%(base_img_id)s, %(overlapping_bounds)s, FALSE, NULL);
    """
    cur.execute(
        query_insert_ms,
        {"base_img_id": base_img_id, "overlapping_bounds": overlapping_ids},
    )

    cur.close()
    conn.close()


def check_multi_stereo_entry(db_name: str, base_image: str) -> bool:
    """
    Check if a multi_stereo entry exists for the given base image.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query = """
        SELECT EXISTS(
            SELECT 1
            FROM multi_stereo ms
            LEFT JOIN image im ON ms.base_img_id = im.img_id
            WHERE im.img_name = %s
        );
    """
    cur.execute(query, (base_image,))
    exists = cur.fetchone()[0]

    cur.close()
    conn.close()
    return bool(exists)


def delete_point_cloud_for_base_image(db_name: str, base_image: str) -> None:
    """
    Delete the point cloud entry associated with a given base image.
    """
    conn = psycopg2.connect(
        user="postgres",
        password="postgres",
        host="127.0.0.1",
        port="5432",
        database=db_name,
    )
    conn.autocommit = True
    cur = conn.cursor()

    query_select_pc = """
        SELECT pc.pc_id
        FROM point_cloud pc
        LEFT JOIN multi_stereo ms
            ON pc.pc_id = ms.pc_id
        LEFT JOIN image im
            ON ms.base_img_id = im.img_id
        WHERE im.img_name = %s;
    """
    cur.execute(query_select_pc, (base_image,))
    row = cur.fetchone()
    if row is None:
        cur.close()
        conn.close()
        return

    pc_id = row[0]
    query_delete_pc = "DELETE FROM point_cloud WHERE pc_id = %s;"
    cur.execute(query_delete_pc, (pc_id,))

    cur.close()
    conn.close()
