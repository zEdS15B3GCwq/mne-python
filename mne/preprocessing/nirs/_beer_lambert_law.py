# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import os.path as op

import numpy as np
from scipy.interpolate import interp1d
from scipy.io import loadmat

from ..._fiff.constants import FIFF
from ...io import BaseRaw
from ...utils import _validate_type, pinv, warn
from ..nirs import _validate_nirs_info, source_detector_distances


def beer_lambert_law(raw, ppf=6.0):
    r"""Convert NIRS optical density data to haemoglobin concentration.

    Parameters
    ----------
    raw : instance of Raw
        The optical density data.
    ppf : tuple | float
        The partial pathlength factors for each wavelength.

        .. versionchanged:: 1.7
           Support for different factors for the two wavelengths.

    Returns
    -------
    raw : instance of Raw
        The modified raw instance.
    """
    raw = raw.copy().load_data()
    _validate_type(raw, BaseRaw, "raw")
    _validate_type(ppf, ("numeric", "array-like"), "ppf")
    picks = _validate_nirs_info(raw.info, fnirs="od", which="Beer-lambert")
    # This is the one place we *really* need the actual/accurate frequencies
    freqs = np.array([raw.info["chs"][pick]["loc"][9] for pick in picks], float)
    ppf = np.array(ppf, float)
    if ppf.ndim == 0:  # upcast single float to (number of freqs, )
        ppf = np.repeat(ppf, len(freqs))
    if ppf.shape != (len(freqs),):
        raise ValueError(
            f"ppf must be float or array-like of shape ({len(freqs),},), got shape {ppf.shape}"
        )
    ppf = ppf[:, np.newaxis]  # shape (len(freqs), 1)
    abs_coef = _load_absorption(freqs)
    distances = source_detector_distances(raw.info, picks="all")
    bad = ~np.isfinite(distances[picks])
    bad |= distances[picks] <= 0
    if bad.any():
        warn(
            "Source-detector distances are zero on NaN, some resulting "
            "concentrations will be zero. Consider setting a montage "
            "with raw.set_montage."
        )
    distances[picks[bad]] = 0.0
    if (distances[picks] > 0.1).any():
        warn(
            "Source-detector distances are greater than 10 cm. "
            "Large distances will result in invalid data, and are "
            "likely due to optode locations being stored in a "
            " unit other than meters."
        )
    rename = dict()

>>>>>>>>>>>>>>>>>>>>>>>>>
    # Number of wavelengths per location
    n_freqs = len(freqs)

    # Total number of measurement points (source-detector pairs)
    n_pairs = len(picks) // n_freqs

    # Process data in groups of n_freqs channels (one group per location)
    for pair_idx in range(n_pairs):
        # Get the channel indices for this pair
        pick_group = picks[pair_idx * n_freqs:(pair_idx + 1) * n_freqs]

        # Get distance (same for all channels at this location)
        dist = distances[pick_group[0]]

        # Calculate the extinction coefficient matrix (wavelengths × chromophores)
        EL = abs_coef * dist * ppf
        iEL = pinv(EL)

        # Apply the conversion to the data
        raw._data[pick_group] = iEL @ raw._data[pick_group] * 1e-3

        # The first two channels in each group are mapped to HbO and HbR
        chromophores = ["hbo", "hbr"]
        coil_dict = {
            "hbo": FIFF.FIFFV_COIL_FNIRS_HBO,
            "hbr": FIFF.FIFFV_COIL_FNIRS_HBR
        }

        # Get original channel name base (without wavelength indicator)
        ch_name_base = raw.info["chs"][pick_group[0]]["ch_name"].split()[0]

        for chrom_idx, kind in enumerate(chromophores):
            if chrom_idx < len(pick_group):  # Ensure we don't exceed available channels
                ch_idx = pick_group[chrom_idx]  # Use first channels for HbO and HbR
                ch = raw.info["chs"][ch_idx]
                ch.update(coil_type=coil_dict[kind], unit=FIFF.FIFF_UNIT_MOL)
                new_name = f"{ch_name_base} {kind}"
                rename[ch["ch_name"]] = new_name

# Old code (original implementation):
"""
    data = list()
    for ii, jj in zip(picks[::2], picks[1::2]):
        dist = distances[ii]
        # Photon path length for each wavelength [m]
        L = dist * ppf
        # Convert to Absorbance change
        a = abs_coef * L
        # Concentration change: (μM)
        c = pinv(a) @ raw._data[[ii, jj]] * 1e-3
        data.append(c)
    data = np.vstack(data)

    # Update info
    for ii, jj in zip(picks[::2], picks[1::2]):
        raw._data[[ii, jj]] = data[:2]
        data = data[2:]

        # Extract source, detector, and wavelength information
        info_ii = raw.info["chs"][ii]
        s_i, d_i, w_i = [info_ii["ch_name"].split()[ii] for ii in range(3)]

        for ch_idx, (idx, kind) in enumerate(
            zip([ii, jj], ["hbo", "hbr"])
        ):
            ch = raw.info["chs"][idx]
            if kind == "hbo":
                ch.update(coil_type=FIFF.FIFFV_COIL_FNIRS_HBO)
            else:
                ch.update(coil_type=FIFF.FIFFV_COIL_FNIRS_HBR)
            ch.update(unit=FIFF.FIFF_UNIT_MOL)
            new_name = f"{s_i} {d_i} {kind}"
            rename[ch["ch_name"]] = new_name
"""
>>>>>>>>>>>>>>>>>>>>>>>>>
    raw.rename_channels(rename)

    # Validate the format of data after transformation is valid
    _validate_nirs_info(raw.info, fnirs="hb")
    return raw


def _load_absorption(freqs: np.ndarray) -> np.ndarray:
    """Load molar extinction coefficients."""
    # Data from https://omlc.org/spectra/hemoglobin/summary.html
    # The text was copied to a text file. The text before and
    # after the table was deleted. The the following was run in
    # matlab
    # extinct_coef=importdata('extinction_coef.txt')
    # save('extinction_coef.mat', 'extinct_coef')
    #
    # Returns data as [[HbO2(freq1), Hb(freq1)],
    #                  [HbO2(freq2), Hb(freq2)]]
    extinction_fname = op.join(
        op.dirname(__file__), "..", "..", "data", "extinction_coef.mat"
    )
    a = loadmat(extinction_fname)["extinct_coef"]

    interp_hbo = interp1d(a[:, 0], a[:, 1], kind="linear")
    interp_hb = interp1d(a[:, 0], a[:, 2], kind="linear")

    ext_coef = np.array([[interp_hbo(freq), interp_hb(freq)] for freq in freqs])
    abs_coef = ext_coef * 0.2303

    return abs_coef
