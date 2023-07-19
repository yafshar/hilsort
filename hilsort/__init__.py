r"""Hilbert-related calculations plus sorting points in Euclidean space using space-filling curves."""
from _hilsort import *

import numpy as np
from typing import List, Union

__all__ = [
    "hilbert_i2c",
    "hilbert_c2i",
    "hilbert_c2i_3_8",
    "hilbert_cmp",
    "hilbert_ieee_cmp",
    "hilbert_box_vtx",
    "hilbert_min_box_vtx",
    "hilbert_max_box_vtx",
    "hilbert_ieee_box_vtx",
    "hilbert_ieee_min_box_vtx",
    "hilbert_ieee_max_box_vtx",
    "hilbert_box_pt",
    "hilbert_min_box_pt",
    "hilbert_max_box_pt",
    "hilbert_ieee_box_pt",
    "hilbert_ieee_min_box_pt",
    "hilbert_ieee_max_box_pt",
    "hilbert_nextinbox",
    "hilbert_incr",
    "hilbert_sort",
    "hilbert_sort_3d",
]


def hilbert_sort(nbits: int, data: Union[List, np.ndarray]) -> np.ndarray:
    """Sort N dimensional (N, ndims) data points based on a Hilbert curve.

    Args:
        nbits (int): Number of bits/coordinate
        data (np.ndarray): data points

    Returns:
        np.ndarray: sorted array of (N, ndims) data points based on a Hilbert curve
    """
    assert nbits >= 1

    data = np.array(data, copy=False)
    if not data.data.c_contiguous:
        data = np.ascontiguousarray(data)

    assert data.ndim == 2, "Input data array should have the shape of (N, ndims)"

    ndims: int = data.shape[1]

    assert (
        ndims * nbits <= 8 * np.dtype(np.uint64).itemsize
    ), f"Product of {ndims} and {nbits} exceeds {8 * np.dtype(np.uint64).itemsize}"

    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    minmax_delta = data_max - data_min
    max_delta = np.max(minmax_delta)
    invBinWidth = (np.left_shift(1, nbits) - 1) / max_delta
    coords = ((data - data_min) * invBinWidth).astype(np.uint64)
    coords = hilbert_sort_bind(ndims, nbits, data, coords)
    return coords


def hilbert_sort_3d(data: Union[List, np.ndarray]) -> np.ndarray:
    """Sort 3D data points based on a Hilbert curve using 8 number of bits/coordinate.

    Args:
        data (np.ndarray): 3D data points (N, 3)

    Returns:
        np.ndarray: sorted array of (N, 3) data points based on a Hilbert curve
    """
    data = np.array(data, copy=False)
    if not data.data.c_contiguous:
        data = np.ascontiguousarray(data)

    assert (
        data.ndim == 2 and data.shape[1] == 3
    ), "Input data array should have the shape of (N, 3)"

    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    minmax_delta = data_max - data_min
    max_delta = np.max(minmax_delta)
    invBinWidth = 255.0 / max_delta
    coords = ((data - data_min) * invBinWidth).astype(np.uint64)
    coords = hilbert_sort_bind_3_8(data, coords)
    return coords


__version__ = "0.0.1"
