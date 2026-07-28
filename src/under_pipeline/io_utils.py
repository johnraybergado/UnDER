from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from osgeo import gdal, gdalconst
from plyfile import PlyData


def write_uint8_tiff(array: np.ndarray, path: str | Path) -> None:
    """
    Write a float image in [0, 1] to a multi-band uint8 GeoTIFF.

    Parameters
    ----------
    array : np.ndarray
        Image array of shape (H, W, C) with values in [0, 1].
    path : str or pathlib.Path
        Output file path for the GeoTIFF.
    """
    rows, cols, nbands = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), cols, rows, nbands, gdalconst.GDT_Byte)
    img_uint8 = (array * 255).astype(np.uint8)
    for i in range(nbands):
        dataset.GetRasterBand(i + 1).WriteArray(img_uint8[:, :, i])
    dataset.FlushCache()


def write_float32_tiff(array: np.ndarray, path: str | Path) -> None:
    """
    Write a float32 multi-band GeoTIFF.

    Parameters
    ----------
    array : np.ndarray
        Image array of shape (H, W, C) stored as float32-compatible data.
    path : str or pathlib.Path
        Output file path for the GeoTIFF.
    """
    rows, cols, nbands = array.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(str(path), cols, rows, nbands, gdalconst.GDT_Float32)
    for i in range(nbands):
        dataset.GetRasterBand(i + 1).WriteArray(array[:, :, i])
    dataset.FlushCache()


def write_ply_basic(
    vertices: np.ndarray,
    colors: np.ndarray,
    path: str | Path,
    pair_idx: int,
) -> None:
    """
    Write or append a basic XYZRGB point cloud to a PLY file.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N, 3) containing XYZ coordinates.
    colors : np.ndarray
        Array of shape (N, 3) containing RGB values in [0, 255] or uint8.
    path : str or pathlib.Path
        Output PLY file path.
    pair_idx : int
        If 0, create a new PLY; otherwise append vertices and update header count.
    """
    import re

    path = Path(path)
    colors = colors.reshape(-1, 3)
    vertices = np.hstack([vertices.reshape(-1, 3), colors])

    if pair_idx == 0:
        header = (
            "ply\n"
            "format ascii 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property uchar red\n"
            "property uchar green\n"
            "property uchar blue\n"
            "end_header\n"
        )
        with path.open("w") as f:
            f.write(header)
            np.savetxt(f, vertices, "%f %f %f %d %d %d")
    else:
        with path.open("r") as f:
            content = f.read()
        match = re.search(r"vertex (\d+)", content)
        if not match:
            raise ValueError("Could not find vertex count in PLY header.")
        current_count = int(match.group(1))
        new_count = current_count + len(vertices)
        content_new = re.sub(r"vertex \d+", f"vertex {new_count}", content)
        with path.open("w") as f:
            f.write(content_new)
            np.savetxt(f, vertices, "%f %f %f %d %d %d")


def write_ply_fpc_format(
    vertices: np.ndarray,
    colors: np.ndarray,
    imgname: str,
    num_images: int,
    get_global_param_fn: Callable[[str], float],
    out_dir: str | Path,
) -> None:
    """
    Write a PLY in the UnDER FPC format (XYZ + RGB + view count) with DB-based shifts.

    Parameters
    ----------
    vertices : np.ndarray
        Array of shape (N, 3) containing XYZ coordinates in world space.
    colors : np.ndarray
        Array of shape (N, 3) containing RGB values in [0, 255] or uint8.
    imgname : str
        Image filename used to derive the PLY filename.
    num_images : int
        Number of source images contributing to each point.
    get_global_param_fn : callable
        Function with signature (param_name) -> value used to fetch x/y shifts.
    out_dir : str or pathlib.Path
        Directory where the PLY file will be written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{Path(imgname).stem}.ply"

    colors = colors.reshape(-1, 3)
    vertices = vertices.reshape(-1, 3)
    nviews = np.full((len(vertices), 1), fill_value=num_images, dtype=np.uint8)

    x_shift = get_global_param_fn("x_shift")
    y_shift = get_global_param_fn("y_shift")
    vertices[:, 0] -= x_shift
    vertices[:, 1] -= y_shift

    vertices = np.hstack([vertices, colors, nviews])

    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar diffuse_red\n"
        "property uchar diffuse_green\n"
        "property uchar diffuse_blue\n"
        "property uchar views\n"
        "end_header\n"
    )

    with path.open("w") as f:
        f.write(header)
        np.savetxt(f, vertices, "%f %f %f %d %d %d %d")


def tile_image(
    array: np.ndarray,
    tile_height: int,
    tile_width: int,
    overlap_height: int,
    overlap_width: int,
) -> np.ndarray:
    """
    Split an image into overlapping tiles of fixed size.

    Parameters
    ----------
    array : np.ndarray
        Image array of shape (H, W) or (H, W, C).
    tile_height : int
        Height of each tile in pixels.
    tile_width : int
        Width of each tile in pixels.
    overlap_height : int
        Vertical overlap between consecutive tiles in pixels.
    overlap_width : int
        Horizontal overlap between consecutive tiles in pixels.

    Returns
    -------
    np.ndarray
        Array of shape (N, tile_height, tile_width, C) containing all tiles.
    """
    if array.ndim == 2:
        nrows, ncols = array.shape
        nbands = 1
        array = np.expand_dims(array, axis=-1)
    else:
        nrows, ncols, nbands = array.shape

    tiles = np.empty((0, tile_height, tile_width, nbands), dtype=np.float64)

    startrow = 0
    endrow = tile_height
    while startrow < nrows:
        startcol = 0
        endcol = tile_width
        while startcol < ncols:
            if (endrow <= nrows) and (endcol <= ncols):
                subset = array[startrow:endrow, startcol:endcol, :]
            elif (endrow > nrows) and (endcol <= ncols):
                subset = array[-tile_height:, startcol:endcol, :]
            elif (endrow > nrows) and (endcol > ncols):
                subset = array[-tile_height:, -tile_width:, :]
            else:
                subset = array[startrow:endrow, -tile_width:, :]

            tiles = np.concatenate(
                (tiles, np.expand_dims(subset, axis=0)), axis=0
            )

            startcol = endcol - overlap_width
            endcol = startcol + tile_width

        startrow = endrow - overlap_height
        endrow = startrow + tile_height

    return tiles


def load_gdal_rgb(path: str | Path) -> np.ndarray:
    """
    Load a GDAL image and return it as an (H, W, C) array.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input raster.

    Returns
    -------
    np.ndarray
        Image array with shape (H, W, C).
    """
    dataset = gdal.Open(str(path))
    array = dataset.ReadAsArray()
    return np.transpose(array, (1, 2, 0))


def load_ply_xyz(path: str | Path) -> np.ndarray:
    """
    Load a PLY file and return an (N, 3) XYZ array.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the input PLY file.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with XYZ coordinates.
    """
    plydata = PlyData.read(str(path))
    vertices = plydata["vertex"].data
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1)
