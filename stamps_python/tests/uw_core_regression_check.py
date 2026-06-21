#!/usr/bin/env python3
"""Regression checks for the StaMPS Step-6 grid translation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
PYTHON_DIR = THIS_DIR.parent
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from uw_core import wrap_filt, uw_grid_wrapped, uw_interp, uw_sb_unwrap_space_time


def _check_signed_edge(value: float, node_a: int, node_b: int, edges: np.ndarray) -> None:
    if node_a == node_b:
        assert value == 0
        return
    assert value != 0
    edge = edges[abs(int(value)) - 1, 1:3]
    np.testing.assert_array_equal(edge, np.sort([node_a, node_b]))
    assert np.sign(value) == (1 if node_a < node_b else -1)


def _check_complex_goldstein_filter() -> None:
    rows, cols = np.indices((48, 48))
    phase = 0.08 * rows + 0.13 * cols + 0.4 * np.sin(rows / 5.0)
    wrapped = np.exp(1j * phase).astype(np.complex64)
    filtered, lowpass = wrap_filt(wrapped, 32, 0.8, "y")

    np.testing.assert_allclose(np.abs(filtered), 1.0, atol=2e-6)
    assert np.mean(np.abs(np.imag(filtered))) > 0.2
    assert lowpass is not None
    assert np.mean(np.abs(np.imag(lowpass))) > 0.2

def main() -> int:
    xy = np.array(
        [
            [1, 0.0, 0.0],
            [2, 200.0, 0.0],
            [3, 600.0, 0.0],
            [4, 0.0, 400.0],
            [5, 400.0, 400.0],
            [6, 600.0, 600.0],
        ]
    )
    day = np.array([-24.0, -12.0, 0.0, 12.0, 24.0])
    master_ix = 2
    ifgday_ix = np.array(
        [[master_ix, i] for i in range(day.size) if i != master_ix],
        dtype=np.int32,
    )
    rate = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5])
    phase = rate[:, None] * day[None, :] / 12.0
    ph = np.exp(1j * phase[:, [0, 1, 3, 4]]).astype(np.complex64)

    uw = uw_grid_wrapped(
        ph,
        xy,
        pix_size=200,
        goldfilt_flag="n",
        lowfilt_flag="n",
    )
    assert (uw.n_i, uw.n_j) == (3, 3)
    assert np.all(uw.grid_ij.min(axis=0) == 0)
    assert np.all(uw.grid_ij.max(axis=0) == 2)

    ui = uw_interp(uw)
    assert ui.rowix.shape == (uw.n_i - 1, uw.n_j)
    assert ui.colix.shape == (uw.n_i, uw.n_j - 1)

    for row in range(ui.Z.shape[0]):
        for col in range(ui.Z.shape[1] - 1):
            _check_signed_edge(
                ui.colix[row, col],
                int(ui.Z[row, col]),
                int(ui.Z[row, col + 1]),
                ui.edgs,
            )
    for row in range(ui.Z.shape[0] - 1):
        for col in range(ui.Z.shape[1]):
            _check_signed_edge(
                ui.rowix[row, col],
                int(ui.Z[row, col]),
                int(ui.Z[row + 1, col]),
                ui.edgs,
            )

    ut = uw_sb_unwrap_space_time(
        uw,
        ui,
        day,
        ifgday_ix,
        np.zeros(ifgday_ix.shape[0]),
        unwrap_method="3D_FULL",
        la_flag="n",
    )
    assert ut.dph_space_uw.shape == (ui.n_edge, ifgday_ix.shape[0])
    assert np.isfinite(ut.dph_space_uw).all()

    # SNAPHU must receive a full nearest-neighbour grid, not sparse zero cells.
    full_grid = uw.ph[ui.Z - 1, 0]
    assert full_grid.shape == uw.nzix.shape
    assert np.all(np.abs(full_grid) > 0)

    print("uw_core regression check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
