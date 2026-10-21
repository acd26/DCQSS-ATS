"""Direct Python translation of the bundled MATLAB CQT_toolbox_2013 path.

Original sources and attribution:
  cqt.m: Copyright (C) 2013 Christian Schörkhuber.
  nsgcqwin.m: Copyright (C) 2013 Nicki Holighaus, Christian Schörkhuber.
  nsgcqwin.m authors: Nicki Holighaus, Gino Velasco, Monika Doerfler;
  edited by Christian Schörkhuber.
  Source documentation: http://nsg.sourceforge.net/doc/generators/nsgcqwin.php

The original sources and this adapted module are licensed under Creative
Commons Attribution-NonCommercial-ShareAlike 3.0 Unported (CC BY-NC-SA 3.0):
https://creativecommons.org/licenses/by-nc-sa/3.0/

Modifications: translated from MATLAB to Python, using the nsgt package for
transform operations and retaining only the forward transform path below.
Only the forward, fully-rasterized, real transform used by ``cqss.m`` is
implemented.  Names and equations follow ``nsgcqwin.m`` and ``cqt.m``.
"""

from __future__ import annotations

from math import ceil

import numpy as np
from nsgt.cq import calcwinrange
from nsgt.nsgtf import nsgtf
from nsgt.util import hannwin


def _matlab_windows(length: int, sample_rate: int, bpo: int, fmin: float, fmax: float):
    nyquist = sample_rate / 2.0
    fmax = min(fmax, nyquist)
    b = int(np.floor(bpo * np.log2(fmax / fmin)))
    positive = fmin * 2.0 ** (np.arange(b + 1) / bpo)
    q_bandwidth = 2 ** (1 / bpo) - 2 ** (-1 / bpo)
    gamma = 228.7 * q_bandwidth
    bandwidth_hz = q_bandwidth * positive + gamma
    positive = positive[positive + bandwidth_hz / 2 <= nyquist]
    bandwidth_hz = bandwidth_hz[: len(positive)]
    count = len(positive)

    centers_hz = np.concatenate(
        ([0.0], positive, [nyquist], sample_rate - positive[::-1])
    )
    positions = np.empty_like(centers_hz, dtype=int)
    positions[: count + 2] = np.floor(centers_hz[: count + 2] * length / sample_rate)
    positions[count + 2 :] = np.ceil(centers_hz[count + 2 :] * length / sample_rate)

    bw_hz = np.concatenate(
        (
            [2 * fmin],
            bandwidth_hz,
            [centers_hz[count + 2] - centers_hz[count]],
            bandwidth_hz[::-1],
        )
    )
    native_m = np.maximum(np.round(bw_hz * length / sample_rate).astype(int), 4)
    windows = [hannwin(int(m)).astype(float) for m in native_m]

    # Tukey-like DC and Nyquist windows from nsgcqwin.m.
    for k in (0, count + 1):
        if native_m[k] > native_m[k + 1]:
            w = np.ones(native_m[k], dtype=float)
            lo = native_m[k] // 2 - native_m[k + 1] // 2
            hi = native_m[k] // 2 + int(ceil(native_m[k + 1] / 2))
            w[lo:hi] = hannwin(int(native_m[k + 1]))
            windows[k] = w / np.sqrt(native_m[k])

    # cqt.m full rasterization: all positive/negative CQT channels use the
    # coefficient count of the highest positive channel.
    raster_m = native_m.copy()
    raster_m[1 : count + 1] = native_m[count]
    raster_m[count + 2 :] = native_m[count:0:-1]

    # cqt.m default 'sine' normalization.
    factors_half = 2.0 * raster_m[: count + 2] / length
    factors = np.concatenate((factors_half, factors_half[-2:0:-1]))
    windows = [window * factor for window, factor in zip(windows, factors)]
    return windows, positions, raster_m, count


def forward_full_real(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    windows, positions, raster_m, count = _matlab_windows(
        len(signal), sample_rate, 60, 50.0, 8000.0
    )
    win_ranges, nn = calcwinrange(windows, positions, len(signal))
    coeffs = nsgtf(
        signal,
        windows,
        win_ranges,
        nn,
        raster_m,
        real=True,
        reducedform=1,
        measurefft=False,
        multithreading=False,
    )
    result = np.asarray(coeffs)
    assert result.shape[0] == count
    return result
