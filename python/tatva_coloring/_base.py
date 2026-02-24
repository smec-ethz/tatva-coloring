"""
Python package wrapper for the `_tatva_coloring` native extension.

Provides lightweight typing hints and user-facing docs when importing
`tatva_coloring` from Python. The compiled extension lives in `_tatva_coloring`.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

try:
    from ._tatva_coloring import (  # ty:ignore[unresolved-import]
        distance2_color_and_seeds as _distance2_color_and_seeds_ext,
    )
    from ._tatva_coloring import (  # ty:ignore[unresolved-import]
        distance2_colors as _distance2_colors_ext,
    )
except ImportError:
    _distance2_color_and_seeds_ext = None
    _distance2_colors_ext = None


__all__ = ["distance2_color_and_seeds", "distance2_colors"]


def distance2_colors(row_ptr: NDArray, col_idx: NDArray, n_dofs: int) -> NDArray:
    """Color the distance-2 graph induced by a CSR sparse matrix.

    Args:
        row_ptr: CSR row pointer of length ``n_dofs + 1``.
        col_idx: CSR column indices.
        n_dofs: Number of rows/cols (degrees of freedom).

    Returns:
        colors: ``np.ndarray`` of shape ``(n_dofs,)`` with color ids (int32).
    """
    if _distance2_colors_ext is None:
        raise ImportError(
            "tatva_color extension is not built. Run `maturin develop` in rust/coloring`."
        )

    row_ptr_arr = np.asarray(row_ptr, dtype=np.int64)
    col_idx_arr = np.asarray(col_idx, dtype=np.int64)
    return _distance2_colors_ext(row_ptr_arr, col_idx_arr, n_dofs)


def distance2_color_and_seeds(
    row_ptr: NDArray, col_idx: NDArray, n_dofs: int
) -> Tuple[NDArray, List[NDArray]]:
    """Color the distance-2 graph induced by a CSR sparse matrix.

    Args:
        row_ptr: CSR row pointer of length ``n_dofs + 1``.
        col_idx: CSR column indices.
        n_dofs: Number of rows/cols (degrees of freedom).

    Returns:
        colors: ``np.ndarray`` of shape ``(n_dofs,)`` with color ids (int32).
        seeds: ``List[np.ndarray]`` of one-hot vectors (bool), one per color.
    """
    if _distance2_color_and_seeds_ext is None:
        raise ImportError(
            "tatva_color extension is not built. Run `maturin develop` in rust/coloring`."
        )

    row_ptr_arr = np.asarray(row_ptr, dtype=np.int64)
    col_idx_arr = np.asarray(col_idx, dtype=np.int64)
    return _distance2_color_and_seeds_ext(row_ptr_arr, col_idx_arr, n_dofs)
