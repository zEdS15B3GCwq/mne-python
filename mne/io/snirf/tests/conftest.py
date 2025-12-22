# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def multi_wavelength_snirf_fname(tmp_path_factory):
    """Return path to a tiny 3-wavelength SNIRF file for io tests."""
    h5py = pytest.importorskip("h5py")
    out_dir = tmp_path_factory.mktemp("snirf_multi")
    fname = Path(out_dir) / "multi_wavelength.snirf"
    if fname.exists():
        return fname

    sfreq = 10.0
    n_times = 32
    time = np.arange(n_times) / sfreq

    # Build two source-detector pairs, each with three wavelengths.
    wavelengths = np.array([700.0, 730.0, 850.0])
    channel_defs = (
        (1, 1, 1),
        (1, 1, 2),
        (1, 1, 3),
        (2, 2, 1),
        (2, 2, 2),
        (2, 2, 3),
    )
    data = np.vstack(
        [
            np.sin(2 * np.pi * (idx + 1) * time / time[-1]) + idx
            for idx in range(len(channel_defs))
        ]
    ).T

    with h5py.File(fname, "w") as f:
        nirs = f.create_group("nirs")
        meta = nirs.create_group("metaDataTags")
        _write_string(meta, "SubjectID", "multi")
        _write_string(meta, "MeasurementDate", "2024-01-01")
        _write_string(meta, "MeasurementTime", "00:00:00")
        _write_string(meta, "LengthUnit", "m")
        _write_string(meta, "TimeUnit", "s")
        _write_string(meta, "FrequencyUnit", "Hz")

        probe = nirs.create_group("probe")
        probe.create_dataset("wavelengths", data=wavelengths)
        probe.create_dataset(
            "sourcePos3D", data=np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0]])
        )
        probe.create_dataset(
            "detectorPos3D", data=np.array([[0.0, 0.03, 0.0], [0.03, 0.03, 0.0]])
        )

        data_group = nirs.create_group("data1")
        data_group.create_dataset("dataTimeSeries", data=data)
        data_group.create_dataset("time", data=time)

        for idx, (src, det, wav) in enumerate(channel_defs, start=1):
            meas = data_group.create_group(f"measurementList{idx}")
            meas.create_dataset("sourceIndex", data=[src])
            meas.create_dataset("detectorIndex", data=[det])
            meas.create_dataset("wavelengthIndex", data=[wav])
            meas.create_dataset("dataType", data=[1])
            meas.create_dataset("dataTypeIndex", data=[1])

    return fname


def _write_string(group, name, value):
    group.create_dataset(name, data=np.array([value.encode("utf-8")]))
