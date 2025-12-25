# Authors: The MNE-Python contributors.
# License: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray


def _build_raw(ch_names, ch_types, freqs=None):
    n_ch = len(ch_names)
    sfreq = 10.0
    n_times = 128
    rng = np.random.default_rng()
    data = rng.random((n_ch, n_times)) + 0.01
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = RawArray(data, info, verbose=True)
    if freqs is not None:
        for ch, freq in zip(raw.info["chs"], freqs):
            ch["loc"][9] = freq
    _assign_optode_positions(raw)
    return raw


def _assign_optode_positions(raw):
    for ch in raw.info["chs"]:
        base, _ = ch["ch_name"].split(" ")
        src, det = base.split("_")
        src_idx = int(src[1:])
        det_idx = int(det[1:])
        ch["loc"][3:6] = np.array([0.01 * src_idx, 0.0, 0.0])
        ch["loc"][6:9] = np.array([0.01 * det_idx, 0.03, 0.0])


@pytest.fixture
def multi_wavelength_raw(request):
    """Create a raw CW fNIRS object with 3 wavelengths per source-detector pair."""
    n_pairs = getattr(request, "param", None)
    if n_pairs is None:
        raise RuntimeError("parametrize multi_wavelength_raw with the desired n_pairs")
    freqs = [700, 730, 850]
    ch_names = [f"S{ii}_D{ii} {wl}" for ii in range(1, n_pairs + 1) for wl in freqs]
    return _build_raw(
        ch_names,
        ["fnirs_cw_amplitude"] * len(ch_names),
        freqs=freqs * n_pairs,
    )
