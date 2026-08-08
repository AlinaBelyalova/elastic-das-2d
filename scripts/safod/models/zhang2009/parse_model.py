#!/usr/bin/env python3
"""
Parse the Zhang, Thurber & Bedrosian (2009) SAFOD tomography files supplied
by Clifford Thurber.

Expected raw files
------------------
MOD.head
inversion_grid.dat
Vp_model.dat
Vs_model.dat
Vpvs_model.dat

The parser follows Cliff Thurber's description:

- MOD.head gives X, Y, Z node coordinates.
- Each model-file line contains all X nodes for one fixed (Y, Z).
- The file advances through Y first, then to the next Z.
- inversion_grid.dat contains one 13-point lat/lon row for each Y node,
  separated by ">" lines.

No interpolation to the SAFOD 2-D modelling plane is performed here.
This script only reconstructs and validates the native 3-D tomography grid.

Outputs
-------
processed/zhang2009_native_grid.npz
processed/zhang2009_native_nodes.csv
qc/zhang2009_grid_summary.txt
qc/zhang2009_xy_grid.png
qc/zhang2009_vp_slices.png
qc/zhang2009_vs_slices.png
qc/zhang2009_vpvs_slices.png

Important
---------
Vp_model.dat and Vs_model.dat are treated as independent primary velocity
models. Vpvs_model.dat is preserved as its own independently inverted field;
it is NOT used to calculate Vs from Vp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_FILES = (
    "MOD.head",
    "inversion_grid.dat",
    "Vp_model.dat",
    "Vs_model.dat",
    "Vpvs_model.dat",
)


def parse_mod_head(path: Path):
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(lines) != 4:
        raise ValueError(
            f"{path} should contain 4 non-empty lines; found {len(lines)}."
        )

    first = lines[0].split()

    if len(first) != 4:
        raise ValueError(
            f"Unexpected first line in {path}: {lines[0]!r}"
        )

    header_scalar = float(first[0])
    nx, ny, nz = map(int, first[1:])

    x_km = np.asarray(
        [float(v) for v in lines[1].split()],
        dtype=np.float64,
    )
    y_km = np.asarray(
        [float(v) for v in lines[2].split()],
        dtype=np.float64,
    )
    z_km = np.asarray(
        [float(v) for v in lines[3].split()],
        dtype=np.float64,
    )

    if len(x_km) != nx:
        raise ValueError(f"nx={nx}, but found {len(x_km)} X nodes.")
    if len(y_km) != ny:
        raise ValueError(f"ny={ny}, but found {len(y_km)} Y nodes.")
    if len(z_km) != nz:
        raise ValueError(f"nz={nz}, but found {len(z_km)} Z nodes.")

    return header_scalar, x_km, y_km, z_km


def parse_inversion_grid(
    path: Path,
    *,
    nx: int,
    ny: int,
):
    groups: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = raw.strip()

        if not text:
            continue

        if text.startswith(">"):
            if current:
                groups.append(current)
                current = []
            continue

        parts = text.split()

        if len(parts) < 2:
            raise ValueError(
                f"{path}:{line_number}: expected LAT LON, got {raw!r}"
            )

        current.append(
            (
                float(parts[0]),
                float(parts[1]),
            )
        )

    if current:
        groups.append(current)

    if len(groups) != ny:
        raise ValueError(
            f"Expected {ny} Y rows in inversion_grid.dat; found {len(groups)}."
        )

    lengths = [len(row) for row in groups]

    if any(length != nx for length in lengths):
        raise ValueError(
            f"Expected {nx} X nodes per inversion-grid row; got {lengths}."
        )

    array = np.asarray(
        groups,
        dtype=np.float64,
    )

    # shape = (ny, nx, 2), values = (lat, lon)
    return array[:, :, 0], array[:, :, 1]


def parse_model_file(
    path: Path,
    *,
    nx: int,
    ny: int,
    nz: int,
):
    rows = []

    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        text = raw.strip()

        if not text:
            continue

        values = [
            float(v)
            for v in text.split()
        ]

        if len(values) != nx:
            raise ValueError(
                f"{path}:{line_number}: expected {nx} X values; "
                f"found {len(values)}."
            )

        rows.append(values)

    expected_rows = ny * nz

    if len(rows) != expected_rows:
        raise ValueError(
            f"{path}: expected {expected_rows} lines (= ny*nz), "
            f"found {len(rows)}."
        )

    rows = np.asarray(
        rows,
        dtype=np.float64,
    )

    # Cliff's ordering:
    #   X varies across columns,
    #   then Y varies down lines,
    #   then the file advances to the next Z.
    return rows.reshape(
        nz,
        ny,
        nx,
    )


def nearest_index(
    values: np.ndarray,
    target: float,
) -> int:
    return int(
        np.argmin(
            np.abs(
                values - target
            )
        )
    )


def plot_xy_grid(
    *,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    output: Path,
):
    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for iy in range(lat_deg.shape[0]):
        ax.plot(
            lon_deg[iy, :],
            lat_deg[iy, :],
            marker=".",
            linewidth=0.8,
        )

    for ix in range(lat_deg.shape[1]):
        ax.plot(
            lon_deg[:, ix],
            lat_deg[:, ix],
            marker=".",
            linewidth=0.8,
        )

    ix0 = nearest_index(
        x_km,
        0.0,
    )
    iy0 = nearest_index(
        y_km,
        0.0,
    )

    ax.scatter(
        [lon_deg[iy0, ix0]],
        [lat_deg[iy0, ix0]],
        s=60,
        marker="x",
        label="Zhang grid X=0, Y=0",
    )

    ax.set_xlabel(
        "Longitude [deg]"
    )
    ax.set_ylabel(
        "Latitude [deg]"
    )
    ax.set_title(
        "Zhang et al. (2009) inversion grid"
    )
    ax.legend()
    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()
    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def plot_slices(
    *,
    cube: np.ndarray,
    x_km: np.ndarray,
    y_km: np.ndarray,
    z_km: np.ndarray,
    quantity: str,
    units: str,
    output: Path,
):
    requested_depths = (
        0.0,
        0.5,
        1.0,
        2.0,
        4.0,
        7.0,
    )

    indices = [
        nearest_index(
            z_km,
            depth,
        )
        for depth in requested_depths
    ]

    # Avoid repeated indices if a future grid is different.
    indices = list(
        dict.fromkeys(
            indices
        )
    )

    ncols = 3
    nrows = int(
        np.ceil(
            len(indices) / ncols
        )
    )

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(15, 4.5 * nrows),
        squeeze=False,
    )

    # Use only the finite local Parkfield volume to set the display scale.
    # The outermost X/Y/Z nodes are artificial far-field bounding nodes.
    local_x = (
        (x_km >= -8.0)
        & (x_km <= 10.0)
    )
    local_y = (
        (y_km >= -8.0)
        & (y_km <= 8.0)
    )
    local_z = (
        (z_km >= 0.0)
        & (z_km <= 7.0)
    )

    finite_values = cube[
        np.ix_(
            local_z,
            local_y,
            local_x,
        )
    ]
    finite_values = finite_values[
        np.isfinite(
            finite_values
        )
    ]

    vmin = float(
        np.percentile(
            finite_values,
            2.0,
        )
    )
    vmax = float(
        np.percentile(
            finite_values,
            98.0,
        )
    )

    x_local = x_km[
        local_x
    ]
    y_local = y_km[
        local_y
    ]

    image = None

    for ax, iz in zip(
        axes.ravel(),
        indices,
    ):
        values = cube[
            iz,
            :,
            :,
        ][
            np.ix_(
                local_y,
                local_x,
            )
        ]

        image = ax.pcolormesh(
            x_local,
            y_local,
            values,
            shading="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(
            f"{quantity}, Z={z_km[iz]:g} km"
        )
        ax.set_xlabel(
            "Zhang X [km]"
        )
        ax.set_ylabel(
            "Zhang Y [km]"
        )

        ax.set_xlim(
            -8.0,
            10.0,
        )
        ax.set_ylim(
            -8.0,
            8.0,
        )
        ax.set_aspect(
            "equal",
            adjustable="box",
        )

    for ax in axes.ravel()[
        len(indices):
    ]:
        ax.set_visible(
            False
        )

    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label=units,
            shrink=0.85,
        )

    fig.suptitle(
        "Native Zhang et al. (2009) tomography — parsing QC"
    )

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(
        fig
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_dir = args.output_dir

    missing = [
        name
        for name in EXPECTED_FILES
        if not (
            raw_dir / name
        ).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing Zhang-2009 files:\n  "
            + "\n  ".join(
                missing
            )
        )

    processed_dir = (
        output_dir
        / "processed"
    )
    qc_dir = (
        output_dir
        / "qc"
    )

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    qc_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        header_scalar,
        x_km,
        y_km,
        z_km,
    ) = parse_mod_head(
        raw_dir / "MOD.head"
    )

    nx = len(x_km)
    ny = len(y_km)
    nz = len(z_km)

    (
        lat_deg,
        lon_deg,
    ) = parse_inversion_grid(
        raw_dir / "inversion_grid.dat",
        nx=nx,
        ny=ny,
    )

    vp_km_s = parse_model_file(
        raw_dir / "Vp_model.dat",
        nx=nx,
        ny=ny,
        nz=nz,
    )

    vs_km_s = parse_model_file(
        raw_dir / "Vs_model.dat",
        nx=nx,
        ny=ny,
        nz=nz,
    )

    vpvs_joint = parse_model_file(
        raw_dir / "Vpvs_model.dat",
        nx=nx,
        ny=ny,
        nz=nz,
    )

    vp_over_vs = (
        vp_km_s
        / vs_km_s
    )

    vpvs_difference = (
        vp_over_vs
        - vpvs_joint
    )

    # Save native grid.
    npz_path = (
        processed_dir
        / "zhang2009_native_grid.npz"
    )

    np.savez_compressed(
        npz_path,
        source=np.array(
            "Zhang, Thurber & Bedrosian (2009), "
            "doi:10.1029/2009GC002709"
        ),
        ordering=np.array(
            "cube[z_index, y_index, x_index]; X fastest in model files"
        ),
        mod_head_scalar=np.array(
            header_scalar,
            dtype=np.float64,
        ),
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        vp_km_s=vp_km_s,
        vs_km_s=vs_km_s,
        vpvs_joint=vpvs_joint,
        vp_over_vs=vp_over_vs,
    )

    # Save a long-form node table for transparent inspection.
    rows = []

    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                rows.append(
                    {
                        "iz": iz,
                        "iy": iy,
                        "ix": ix,
                        "x_km": x_km[ix],
                        "y_km": y_km[iy],
                        "z_km": z_km[iz],
                        "latitude_deg": lat_deg[iy, ix],
                        "longitude_deg": lon_deg[iy, ix],
                        "vp_km_s": vp_km_s[iz, iy, ix],
                        "vs_km_s": vs_km_s[iz, iy, ix],
                        "vpvs_joint": vpvs_joint[iz, iy, ix],
                        "vp_div_vs": vp_over_vs[iz, iy, ix],
                    }
                )

    csv_path = (
        processed_dir
        / "zhang2009_native_nodes.csv"
    )

    pd.DataFrame(
        rows
    ).to_csv(
        csv_path,
        index=False,
    )

    ix0 = nearest_index(
        x_km,
        0.0,
    )
    iy0 = nearest_index(
        y_km,
        0.0,
    )

    summary = "\n".join(
        [
            "Zhang, Thurber & Bedrosian (2009) native-model parsing QC",
            "========================================================",
            "",
            f"MOD.head scalar       : {header_scalar}",
            f"grid dimensions       : nx={nx}, ny={ny}, nz={nz}",
            f"native cube shape     : {(nz, ny, nx)} = (z, y, x)",
            f"model values/file     : {nx * ny * nz}",
            "",
            f"X nodes [km]          : {x_km.tolist()}",
            f"Y nodes [km]          : {y_km.tolist()}",
            f"Z nodes [km]          : {z_km.tolist()}",
            "",
            "Grid origin from inversion_grid.dat:",
            f"  lat(X=0,Y=0)        : {lat_deg[iy0, ix0]:.12f}",
            f"  lon(X=0,Y=0)        : {lon_deg[iy0, ix0]:.12f}",
            "",
            f"Vp range [km/s]       : {np.nanmin(vp_km_s):.3f} .. {np.nanmax(vp_km_s):.3f}",
            f"Vs range [km/s]       : {np.nanmin(vs_km_s):.3f} .. {np.nanmax(vs_km_s):.3f}",
            f"joint Vp/Vs range     : {np.nanmin(vpvs_joint):.3f} .. {np.nanmax(vpvs_joint):.3f}",
            "",
            "Independent Vp/Vs QC:",
            f"  median |Vp/Vs-joint|: {np.nanmedian(np.abs(vpvs_difference)):.6f}",
            f"  p95    |Vp/Vs-joint|: {np.nanpercentile(np.abs(vpvs_difference), 95):.6f}",
            f"  max    |Vp/Vs-joint|: {np.nanmax(np.abs(vpvs_difference)):.6f}",
            "",
            "Interpretation:",
            "  Vp_model.dat and Vs_model.dat are retained independently.",
            "  Vpvs_model.dat is retained as a separate jointly inverted field.",
            "  No 2-D section extraction or interpolation has been performed.",
            "",
        ]
    )

    summary_path = (
        qc_dir
        / "zhang2009_grid_summary.txt"
    )

    summary_path.write_text(
        summary,
        encoding="utf-8",
    )

    plot_xy_grid(
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        x_km=x_km,
        y_km=y_km,
        output=(
            qc_dir
            / "zhang2009_xy_grid.png"
        ),
    )

    plot_slices(
        cube=vp_km_s,
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        quantity="Vp",
        units="km/s",
        output=(
            qc_dir
            / "zhang2009_vp_slices.png"
        ),
    )

    plot_slices(
        cube=vs_km_s,
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        quantity="Vs",
        units="km/s",
        output=(
            qc_dir
            / "zhang2009_vs_slices.png"
        ),
    )

    plot_slices(
        cube=vpvs_joint,
        x_km=x_km,
        y_km=y_km,
        z_km=z_km,
        quantity="Vp/Vs",
        units="ratio",
        output=(
            qc_dir
            / "zhang2009_vpvs_slices.png"
        ),
    )

    print(summary)
    print("Saved:")
    print(f"  {npz_path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {qc_dir / 'zhang2009_xy_grid.png'}")
    print(f"  {qc_dir / 'zhang2009_vp_slices.png'}")
    print(f"  {qc_dir / 'zhang2009_vs_slices.png'}")
    print(f"  {qc_dir / 'zhang2009_vpvs_slices.png'}")


if __name__ == "__main__":
    main()
