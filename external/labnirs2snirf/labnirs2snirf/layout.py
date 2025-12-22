"""
Functions related to importing probe positions.
"""

import logging
from pathlib import Path

from .error import Labnirs2SnirfError
from .model import Nirs


class LayoutError(Labnirs2SnirfError):
    """Custom error class for layout-related issues."""


type Layout_3D = dict[str, tuple[float, float, float]]

log = logging.getLogger(__name__)


def read_layout(file: Path) -> Layout_3D:
    """
    Read optode coordinates from file.

    Parameters
    ----------
    file : Path
        Path pointing to location file.

    Returns
    -------
    dict[str, tuple[float, float, float]]
        Dict mapping probe labels to 3D coordinates.

    Notes
    -----
    Location files are expected to follow the .sfp format, which is a tab-separated text file
    with columns: label, x, y, and z, where 'x', 'y', and 'z' are the 3D coordinates of the optode.
    Labels are case-sensitive. If duplicate labels are found, a warning is logged and the last
    occurrence is used.
    """

    import polars as pl  # pylint: disable=C0415

    log.debug("Reading layout file: %s", file)

    try:
        locations = (
            pl.scan_csv(
                file,
                has_header=False,
                separator="\t",
                schema=pl.Schema(
                    zip(["label", "x", "y", "z"], [pl.String] + [pl.Float64] * 3),
                ),
            )
            # remove whitespace around values
            .with_columns(pl.col(pl.String).str.strip_chars())
            .collect()
        )

        # Check for duplicate labels
        if locations.height != locations.select("label").unique().height:
            duplicates = (
                locations.group_by("label")
                .agg(pl.len().alias("count"))
                .filter(pl.col("count") > 1)
                .get_column("label")
                .to_list()
            )
            log.warning(
                "Duplicate labels found in layout file: %s. Last occurrence will be used.",
                ", ".join(duplicates),
            )

        layout_dict = {
            row["label"]: (row["x"], row["y"], row["z"])
            for row in locations.rows(named=True)
        }

        log.debug(
            "Successfully read %d optode positions from layout file",
            len(layout_dict),
        )
        return layout_dict

    except (
        pl.exceptions.ComputeError,
        pl.exceptions.ColumnNotFoundError,
        pl.exceptions.SchemaError,
    ) as e:
        log.exception("Failed to read layout file %s: %s", file, e)
        raise LayoutError(
            f"Failed to read layout file {file}: {e}. "
            "Ensure the file is tab-separated with four columns: label, x, y, z,"
            " and that x, y, z are numeric values.",
        ) from e
    except FileNotFoundError as e:
        log.exception("Layout file not found: %s", file)
        raise LayoutError(f"Layout file not found: {file}") from e

    except Exception as ex:
        log.exception("Failed to read layout file %s: %s", file, ex)
        raise


def update_layout(data: Nirs, locations: Layout_3D) -> None:
    """
    Update source and position 3D coordinates in nirs data.

    Parameters
    ----------
    data : model.Nirs
        Nirs data produced by `labirs.read_nirs()`. Probes have labels (Si, Di)
        corresponding to their original numbering in the exported labNIRS data.
    locations : dict[str, tuple[float, float, float]]
        Mapping of probe labels to 3D coordinates. These can be read in from file
        by `read_layout()`. Coordinates are always stored as 3D tuples - when only
        2D is available, Z is set to 0. Labels are case-sensitive.

    Raises
    ------
    LayoutError
        If no probe labels present in ``data``.
    """
    if data.probe.sourceLabels is None and data.probe.detectorLabels is None:
        raise LayoutError(
            "Update layout failed because there are no probe labels in NIRS data.",
        )

    log.debug("Updating probe positions with %d provided locations", len(locations))

    sources_updated = 0
    detectors_updated = 0

    # Update source positions
    if data.probe.sourceLabels:
        for i, label in enumerate(data.probe.sourceLabels):
            if label in locations:
                data.probe.sourcePos3D[i, :] = locations[label]
                sources_updated += 1
            else:
                log.warning(
                    "Source %s missing position data, keeping default (0, 0, 0)",
                    label,
                )

    # Update detector positions
    if data.probe.detectorLabels:
        for i, label in enumerate(data.probe.detectorLabels):
            if label in locations:
                data.probe.detectorPos3D[i, :] = locations[label]
                detectors_updated += 1
            else:
                log.warning(
                    "Detector %s missing position data, keeping default (0, 0, 0)",
                    label,
                )

    log.info(
        "Updated positions for %d sources and %d detectors",
        sources_updated,
        detectors_updated,
    )
