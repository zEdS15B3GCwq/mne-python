# Authors: The MNE-Python contributors.
# License: BSD-3-Clause
# Copyright the MNE-Python contributors.

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_almost_equal, assert_array_equal

from mne import create_info
from mne._fiff.constants import FIFF
from mne._fiff.pick import _picks_to_idx
from mne.datasets import testing
from mne.datasets.testing import data_path
from mne.io import read_raw_nirx
from mne.preprocessing.nirs import (
    _channel_chromophore,
    _channel_frequencies,
    _check_channels_ordered,
    _fnirs_optode_names,
    _fnirs_spread_bads,
    _optode_position,
    _validate_nirs_info,
    beer_lambert_law,
    optical_density,
    scalp_coupling_index,
    tddr,
)

fname_nirx_15_0 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_0_recording"
)
fname_nirx_15_2 = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording"
)
fname_nirx_15_2_short = (
    data_path(download=False) / "NIRx" / "nirscout" / "nirx_15_2_recording_w_short"
)
_SYNTHETIC_MULTI = "synthetic_multi"


@testing.requires_testing_data
def test_fnirs_picks():
    """Test picking of fnirs types after different conversions."""
    raw = read_raw_nirx(fname_nirx_15_0)
    picks = _picks_to_idx(raw.info, "fnirs_cw_amplitude")
    assert len(picks) == len(raw.ch_names)
    raw_subset = raw.copy().pick(picks="fnirs_cw_amplitude")
    for ch in raw_subset.info["chs"]:
        assert ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_CW_AMPLITUDE

    picks = _picks_to_idx(raw.info, ["fnirs_cw_amplitude", "fnirs_od"])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ["fnirs_cw_amplitude", "fnirs_od", "hbr"])
    assert len(picks) == len(raw.ch_names)
    pytest.raises(ValueError, _picks_to_idx, raw.info, "fnirs_od")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "hbo")
    pytest.raises(ValueError, _picks_to_idx, raw.info, ["hbr"])
    pytest.raises(ValueError, _picks_to_idx, raw.info, "fnirs_fd_phase")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "junk")

    raw = optical_density(raw)
    picks = _picks_to_idx(raw.info, "fnirs_od")
    assert len(picks) == len(raw.ch_names)
    raw_subset = raw.copy().pick(picks="fnirs_od")
    for ch in raw_subset.info["chs"]:
        assert ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_OD

    picks = _picks_to_idx(raw.info, ["fnirs_cw_amplitude", "fnirs_od"])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ["fnirs_cw_amplitude", "fnirs_od", "hbr"])
    assert len(picks) == len(raw.ch_names)
    pytest.raises(ValueError, _picks_to_idx, raw.info, "fnirs_cw_amplitude")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "hbo")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "hbr")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "fnirs_fd_phase")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "junk")

    raw = beer_lambert_law(raw)
    picks = _picks_to_idx(raw.info, "hbo")
    assert len(picks) == len(raw.ch_names) / 2
    raw_subset = raw.copy().pick(picks="hbo")
    for ch in raw_subset.info["chs"]:
        assert ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_HBO

    picks = _picks_to_idx(raw.info, ["hbr"])
    assert len(picks) == len(raw.ch_names) / 2
    raw_subset = raw.copy().pick(picks=["hbr"])
    for ch in raw_subset.info["chs"]:
        assert ch["coil_type"] == FIFF.FIFFV_COIL_FNIRS_HBR

    picks = _picks_to_idx(raw.info, ["hbo", "hbr"])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ["hbo", "fnirs_od", "hbr"])
    assert len(picks) == len(raw.ch_names)
    picks = _picks_to_idx(raw.info, ["hbo", "fnirs_od"])
    assert len(picks) == len(raw.ch_names) / 2
    pytest.raises(ValueError, _picks_to_idx, raw.info, "fnirs_cw_amplitude")
    pytest.raises(ValueError, _picks_to_idx, raw.info, ["fnirs_od"])
    pytest.raises(ValueError, _picks_to_idx, raw.info, "junk")
    pytest.raises(ValueError, _picks_to_idx, raw.info, "fnirs_fd_phase")


# Backward compat wrapper for simplicity below
def _fnirs_check_bads(info):
    _validate_nirs_info(info)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname", ([fname_nirx_15_2_short, fname_nirx_15_2, fname_nirx_15_0])
)
def test_fnirs_check_bads(fname):
    """Test checking of bad markings."""
    # No bad channels, so these should all pass
    raw = read_raw_nirx(fname)
    _fnirs_check_bads(raw.info)
    raw = optical_density(raw)
    _fnirs_check_bads(raw.info)
    raw = beer_lambert_law(raw)
    _fnirs_check_bads(raw.info)

    # Mark pairs of bad channels, so these should all pass
    raw = read_raw_nirx(fname)
    raw.info["bads"] = raw.ch_names[0:2]
    _fnirs_check_bads(raw.info)
    raw = optical_density(raw)
    _fnirs_check_bads(raw.info)
    raw = beer_lambert_law(raw)
    _fnirs_check_bads(raw.info)

    # Mark single channel as bad, so these should all fail
    raw = read_raw_nirx(fname)
    raw.info["bads"] = raw.ch_names[0:1]
    pytest.raises(RuntimeError, _fnirs_check_bads, raw.info)
    with pytest.raises(RuntimeError, match="bad labelling"):
        raw = optical_density(raw)
    raw.info["bads"] = []
    raw = optical_density(raw)
    raw.info["bads"] = raw.ch_names[0:1]
    pytest.raises(RuntimeError, _fnirs_check_bads, raw.info)
    with pytest.raises(RuntimeError, match="bad labelling"):
        raw = beer_lambert_law(raw)
    pytest.raises(RuntimeError, _fnirs_check_bads, raw.info)


@testing.requires_testing_data
@pytest.mark.parametrize(
    "fname", ([fname_nirx_15_2_short, fname_nirx_15_2, fname_nirx_15_0])
)
def test_fnirs_spread_bads(fname):
    """Test checking of bad markings."""
    # Test spreading upwards in frequency and on raw data
    raw = read_raw_nirx(fname)
    raw.info["bads"] = ["S1_D1 760"]
    info = _fnirs_spread_bads(raw.info)
    assert info["bads"] == ["S1_D1 760", "S1_D1 850"]

    # Test spreading downwards in frequency and on od data
    raw = optical_density(raw)
    raw.info["bads"] = raw.ch_names[5:6]
    info = _fnirs_spread_bads(raw.info)
    assert info["bads"] == raw.ch_names[4:6]

    # Test spreading multiple bads and on chroma data
    raw = beer_lambert_law(raw)
    raw.info["bads"] = [raw.ch_names[x] for x in [1, 8]]
    info = _fnirs_spread_bads(raw.info)
    assert info["bads"] == [info.ch_names[x] for x in [0, 1, 8, 9]]


@testing.requires_testing_data
@pytest.mark.parametrize(
    "dataset",
    (
        pytest.param(fname_nirx_15_2_short, id="nirx_15_2_short"),
        pytest.param(fname_nirx_15_2, id="nirx_15_2"),
        pytest.param(fname_nirx_15_0, id="nirx_15_0"),
        pytest.param(_SYNTHETIC_MULTI, id=_SYNTHETIC_MULTI),
    ),
)
def test_fnirs_channel_naming_and_order_readers(dataset, multi_wavelength_raw):
    """Ensure fNIRS channel checking on standard readers."""
    if dataset == _SYNTHETIC_MULTI:
        raw = multi_wavelength_raw.copy()
    else:
        raw = read_raw_nirx(dataset)

    freqs = np.unique(_channel_frequencies(raw.info))
    if len(freqs) == 2:
        assert_array_equal(freqs, [760, 850])
    else:
        assert_array_equal(freqs, [700, 730, 850])
    chroma = np.unique(_channel_chromophore(raw.info))
    assert len(chroma) == 0

    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)

    raw_dropped = raw.copy().drop_channels(raw.ch_names[4])
    with pytest.raises(ValueError, match="not ordered correctly"):
        _check_channels_ordered(raw_dropped.info, freqs)

    if len(freqs) == 2:
        raw_names_reversed = raw.copy().ch_names
        raw_names_reversed.reverse()
        raw_reversed = raw.copy().pick(raw_names_reversed)
        with pytest.raises(ValueError, match="The frequencies.*sorted.*"):
            _check_channels_ordered(raw_reversed.info, [850, 760])
        picks = _check_channels_ordered(raw_reversed.info, freqs)
        got_first = set(raw_reversed.ch_names[pick].split()[1] for pick in picks[::2])
        assert got_first == {"760"}
        got_second = set(raw_reversed.ch_names[pick].split()[1] for pick in picks[1::2])
        assert got_second == {"850"}

    raw = optical_density(raw)
    freqs = np.unique(_channel_frequencies(raw.info))
    if len(freqs) == 2:
        assert_array_equal(freqs, [760, 850])
    else:
        assert_array_equal(freqs, [700, 730, 850])
    chroma = np.unique(_channel_chromophore(raw.info))
    assert len(chroma) == 0
    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)

    raw = beer_lambert_law(raw)
    freqs = np.unique(_channel_frequencies(raw.info))
    assert len(freqs) == 0
    assert len(_channel_chromophore(raw.info)) == len(raw.ch_names)
    chroma = np.unique(_channel_chromophore(raw.info))
    assert_array_equal(chroma, ["hbo", "hbr"])
    picks = _check_channels_ordered(raw.info, chroma)
    assert len(picks) == len(raw.ch_names)
    with pytest.raises(ValueError, match="chromophore in info"):
        _check_channels_ordered(raw.info, ["hbr", "hbo"])


def test_fnirs_channel_naming_and_order_custom_raw(
    fnirs_base_cw_raw, multi_wavelength_raw
):
    """Ensure fNIRS channel checking on manually created data."""
    raw = fnirs_base_cw_raw.copy()
    freqs = np.unique(_channel_frequencies(raw.info))
    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)

    # Different systems use different wavelengths
    raw_alt = fnirs_base_cw_raw.copy()
    alt_freqs = np.tile([920.0, 850.0], 3)
    rename = {}
    for idx, ch_name in enumerate(raw_alt.ch_names):
        base = ch_name.split()[0]
        rename[ch_name] = f"{base} {int(alt_freqs[idx])}"
        raw_alt.info["chs"][idx]["loc"][9] = alt_freqs[idx]
    raw_alt.rename_channels(rename)
    picks = _check_channels_ordered(raw_alt.info, [850, 920])
    assert len(picks) == len(raw_alt.ch_names)

    raw_mismatch = fnirs_base_cw_raw.copy()
    raw_mismatch.info["chs"][0]["loc"][9] = 920.0
    with pytest.raises(ValueError, match="not ordered"):
        _check_channels_ordered(raw_mismatch.info, [850, 920])

    raw_missing = fnirs_base_cw_raw.copy()
    for ch in raw_missing.info["chs"]:
        ch["loc"][9] = 0.0
    with pytest.raises(ValueError, match="missing wavelength information"):
        _check_channels_ordered(raw_missing.info, [850, 920])

    raw_block = fnirs_base_cw_raw.copy()
    block_names = [
        "S1_D1 760",
        "S2_D1 760",
        "S3_D1 760",
        "S1_D1 850",
        "S2_D1 850",
        "S3_D1 850",
    ]
    raw_block.rename_channels(dict(zip(raw_block.ch_names, block_names)))
    block_freqs = np.repeat([760, 850], 3)
    for ch, freq in zip(raw_block.info["chs"], block_freqs):
        ch["loc"][9] = freq
    _check_channels_ordered(raw_block.info, [760, 850])
    raw_block.pick(picks=[0, 3, 1, 4, 2, 5])
    _check_channels_ordered(raw_block.info, [760, 850])

    raw_multi = multi_wavelength_raw.copy()
    freqs_multi = np.unique(_channel_frequencies(raw_multi.info))
    picks = _check_channels_ordered(raw_multi.info, freqs_multi)
    assert len(picks) == len(raw_multi.ch_names)
    with pytest.raises(ValueError, match="sorted"):
        _check_channels_ordered(raw_multi.info, freqs_multi[::-1])


def test_fnirs_channel_naming_and_order_custom_optical_density(
    fnirs_base_od_raw, fnirs_base_chroma_raw
):
    """Ensure fNIRS channel checking on manually created data."""
    raw = fnirs_base_od_raw.copy()
    freqs = np.unique(_channel_frequencies(raw.info))
    picks = _check_channels_ordered(raw.info, freqs)
    assert len(picks) == len(raw.ch_names)

    raw_block = fnirs_base_od_raw.copy()
    block_names = [
        "S1_D1 760",
        "S2_D1 760",
        "S3_D1 760",
        "S1_D1 850",
        "S2_D1 850",
        "S3_D1 850",
    ]
    raw_block.rename_channels(dict(zip(raw_block.ch_names, block_names)))
    block_freqs = np.repeat([760, 850], 3)
    for ch, freq in zip(raw_block.info["chs"], block_freqs):
        ch["loc"][9] = freq
    _check_channels_ordered(raw_block.info, [760, 850])
    raw_block.pick(picks=[0, 3, 1, 4, 2, 5])
    _check_channels_ordered(raw_block.info, [760, 850])

    raw_mixed = fnirs_base_od_raw.copy()
    raw_mixed.add_channels([fnirs_base_chroma_raw.copy()])
    with pytest.raises(ValueError, match="does not support a combination"):
        _check_channels_ordered(raw_mixed.info, [760, 850])


def test_fnirs_channel_naming_and_order_custom_chroma(fnirs_base_chroma_raw):
    """Ensure fNIRS channel checking on manually created data."""
    raw = fnirs_base_chroma_raw.copy()
    chroma = np.unique(_channel_chromophore(raw.info))
    picks = _check_channels_ordered(raw.info, chroma)
    assert len(picks) == len(raw.ch_names)

    raw_block = fnirs_base_chroma_raw.copy()
    block_names = [
        "S1_D1 hbo",
        "S2_D1 hbo",
        "S3_D1 hbo",
        "S1_D1 hbr",
        "S2_D1 hbr",
        "S3_D1 hbr",
    ]
    raw_block.rename_channels(dict(zip(raw_block.ch_names, block_names)))
    _check_channels_ordered(raw_block.info, ["hbo", "hbr"])
    raw_block.pick(picks=[0, 3, 1, 4, 2, 5])
    _check_channels_ordered(raw_block.info, ["hbo", "hbr"])
    with pytest.raises(ValueError, match="chromophore in info"):
        _check_channels_ordered(raw_block.info, ["hbb", "hbr"])

    raw_bad_name = fnirs_base_chroma_raw.copy()
    rename = {name: name.replace("hbo", "hbb") for name in raw_bad_name.ch_names}
    raw_bad_name.rename_channels(rename)
    with pytest.raises(ValueError, match="naming conventions"):
        _check_channels_ordered(raw_bad_name.info, ["hbo", "hbr"])

    raw_bad_det = fnirs_base_chroma_raw.copy()
    rename = {}
    for name in raw_bad_det.ch_names:
        base, chrom = name.split()
        src, det = base.split("_")
        if src == "S1":
            rename[name] = f"S1_DX {chrom}"
    raw_bad_det.rename_channels(rename)
    with pytest.raises(ValueError, match="can not be parsed"):
        _check_channels_ordered(raw_bad_det.info, ["hbo", "hbr"])


def test_optode_names():
    """Ensure optode name extraction is correct."""
    ch_names = [
        "S11_D2 760",
        "S11_D2 850",
        "S3_D1 760",
        "S3_D1 850",
        "S2_D13 760",
        "S2_D13 850",
    ]
    ch_types = np.repeat("fnirs_od", 6)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    src_names, det_names = _fnirs_optode_names(info)
    assert_array_equal(src_names, [f"S{n}" for n in ["2", "3", "11"]])
    assert_array_equal(det_names, [f"D{n}" for n in ["1", "2", "13"]])

    ch_names = [
        "S1_D11 hbo",
        "S1_D11 hbr",
        "S2_D17 hbo",
        "S2_D17 hbr",
        "S3_D1 hbo",
        "S3_D1 hbr",
    ]
    ch_types = np.tile(["hbo", "hbr"], 3)
    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=1.0)
    src_names, det_names = _fnirs_optode_names(info)
    assert_array_equal(src_names, [f"S{n}" for n in range(1, 4)])
    assert_array_equal(det_names, [f"D{n}" for n in ["1", "11", "17"]])


@testing.requires_testing_data
def test_optode_loc():
    """Ensure optode location extraction is correct."""
    raw = read_raw_nirx(fname_nirx_15_2_short)
    loc = _optode_position(raw.info, "D3")
    assert_array_almost_equal(loc, [0.082804, 0.01573, 0.024852])


def test_order_agnostic(nirx_snirf):
    """Test that order does not matter to (pre)processing results."""
    raw_nirx, raw_snirf = nirx_snirf
    raw_random = raw_nirx.copy().pick(
        np.random.RandomState(0).permutation(len(raw_nirx.ch_names))
    )
    raws = dict(nirx=raw_nirx, snirf=raw_snirf, random=raw_random)
    del raw_nirx, raw_snirf, raw_random
    orders = dict()
    # continuous wave
    for key, r in raws.items():
        assert set(r.get_channel_types()) == {"fnirs_cw_amplitude"}
        orders[key] = [r.ch_names.index(name) for name in raws["nirx"].ch_names]
        assert_array_equal(raws["nirx"].ch_names, np.array(r.ch_names)[orders[key]])
        assert_allclose(raws["nirx"].get_data(), r.get_data(orders[key]), err_msg=key)
    assert_array_equal(orders["nirx"], np.arange(len(raws["nirx"].ch_names)))
    # optical density
    for key, r in raws.items():
        raws[key] = r = optical_density(r)
        assert_allclose(raws["nirx"].get_data(), r.get_data(orders[key]), err_msg=key)
        assert set(r.get_channel_types()) == {"fnirs_od"}
    # scalp-coupling index
    sci = dict()
    for key, r in raws.items():
        sci[key] = r = scalp_coupling_index(r)
        assert_allclose(sci["nirx"], r[orders[key]], err_msg=key, rtol=0.01)
    # TDDR (on optical)
    tddrs = dict()
    for key, r in raws.items():
        tddrs[key] = r = tddr(r)
        assert_allclose(
            tddrs["nirx"].get_data(), r.get_data(orders[key]), err_msg=key, atol=1e-4
        )
        assert set(r.get_channel_types()) == {"fnirs_od"}
    # beer-lambert
    for key, r in raws.items():
        raws[key] = r = beer_lambert_law(r)
        assert_allclose(
            raws["nirx"].get_data(), r.get_data(orders[key]), err_msg=key, rtol=2e-7
        )
        assert set(r.get_channel_types()) == {"hbo", "hbr"}
    # TDDR (on haemo)
    tddrs = dict()
    for key, r in raws.items():
        tddrs[key] = r = tddr(r)
        assert_allclose(
            tddrs["nirx"].get_data(), r.get_data(orders[key]), err_msg=key, atol=1e-9
        )
        assert set(r.get_channel_types()) == {"hbo", "hbr"}
