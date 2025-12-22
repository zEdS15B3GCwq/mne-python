# Authors: The MNE-Python contributors.
# License: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest

from mne import create_info
from mne.io import RawArray


def _build_raw(ch_names, ch_types, freqs=None, *, positive=False):
    n_ch = len(ch_names)
    n_times = 64
    sfreq = 10.0
    rng = np.random.RandomState(0)
    data = rng.randn(n_ch, n_times)
    if positive:
        data = np.abs(data) + 0.1
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
        ch["loc"][3:6] = np.array([0.03 * src_idx, 0.0, 0.0])
        ch["loc"][6:9] = np.array([0.03 * det_idx, 0.03, 0.0])


@pytest.fixture
def fnirs_base_cw_raw():
    """Return cw-amplitude Raw with three source-detector pairs."""
    ch_names = [
        "S1_D1 760",
        "S1_D1 850",
        "S2_D1 760",
        "S2_D1 850",
        "S3_D1 760",
        "S3_D1 850",
    ]
    freqs = np.tile([760.0, 850.0], 3)
    return _build_raw(
        ch_names,
        ["fnirs_cw_amplitude"] * len(ch_names),
        freqs=freqs,
        positive=True,
    )


@pytest.fixture
def fnirs_base_od_raw():
    """Return optical-density Raw with three source-detector pairs."""
    ch_names = [
        "S1_D1 760",
        "S1_D1 850",
        "S2_D1 760",
        "S2_D1 850",
        "S3_D1 760",
        "S3_D1 850",
    ]
    freqs = np.tile([760.0, 850.0], 3)
    return _build_raw(ch_names, ["fnirs_od"] * len(ch_names), freqs=freqs)


@pytest.fixture
def fnirs_base_chroma_raw():
    """Return haemoglobin Raw with alternating HbO/HbR channels."""
    ch_names = [
        "S1_D1 hbo",
        "S1_D1 hbr",
        "S2_D1 hbo",
        "S2_D1 hbr",
        "S3_D1 hbo",
        "S3_D1 hbr",
    ]
    return _build_raw(ch_names, list(np.tile(["hbo", "hbr"], 3)))


@pytest.fixture
def multi_wavelength_raw():
    """Return cw Raw with three wavelengths per source-detector pair."""
    ch_names = [
        "S1_D1 700",
        "S1_D1 730",
        "S1_D1 850",
        "S2_D2 700",
        "S2_D2 730",
        "S2_D2 850",
    ]
    freqs = [700.0, 730.0, 850.0] * 2
    return _build_raw(
        ch_names,
        ["fnirs_cw_amplitude"] * len(ch_names),
        freqs=freqs,
        positive=True,
    )
