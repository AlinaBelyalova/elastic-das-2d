from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.safod_deep.catalog import (
    H5RecordingScanner,
    RecordingFile,
    RecordingRoot,
)


# ------------------------------------------------------------------------------
# Worker state
# ------------------------------------------------------------------------------

_SCANNER: H5RecordingScanner | None = None
_ROOT: RecordingRoot | None = None


def _init_worker(
    label: str,
    root_path: str,
    priority: int,
    notes: str,
) -> None:
    global _SCANNER, _ROOT

    _SCANNER = H5RecordingScanner()
    _ROOT = RecordingRoot(
        label=label,
        path=root_path,
        priority=priority,
        notes=notes,
    )


def _scan_one(path_str: str) -> dict[str, Any]:
    if _SCANNER is None or _ROOT is None:
        raise RuntimeError("Worker was not initialized.")

    record = _SCANNER.scan_file(
        Path(path_str),
        _ROOT,
    )

    return asdict(record)


# ------------------------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------------------------

COLUMNS = list(
    RecordingFile.__dataclass_fields__.keys()
)


def _make_frame(
    rows: list[dict[str, Any]],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=COLUMNS,
    )

    if frame.empty:
        return frame

    sort_time = pd.to_datetime(
        frame["start_time_utc"],
        utc=True,
        errors="coerce",
    )

    frame = (
        frame.assign(_sort_time=sort_time)
        .sort_values(
            [
                "_sort_time",
                "file_path",
            ],
            na_position="last",
        )
        .drop(columns="_sort_time")
        .reset_index(drop=True)
    )

    return frame


def _atomic_write(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    frame = _make_frame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    frame.to_csv(
        tmp_path,
        index=False,
    )

    os.replace(
        tmp_path,
        output_path,
    )


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/oak/stanford/groups/ettore88/data/"
            "SAFOD/SAFOD-deep/2026-06-Nano"
        ),
    )

    parser.add_argument(
        "--label",
        default="2026-06_nano",
    )

    parser.add_argument(
        "--priority",
        type=int,
        default=80,
    )

    parser.add_argument(
        "--notes",
        default=(
            "Nano HDF5, June-August 2026; "
            "processed by Ettore."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/safod_deep/catalog/"
            "recording_manifest_nano.csv"
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=min(
            8,
            int(
                os.environ.get(
                    "SLURM_CPUS_PER_TASK",
                    "8",
                )
            ),
        ),
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="For smoke testing only.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Discard an existing checkpoint/output and start again.",
    )

    return parser.parse_args()


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not args.root.exists():
        raise FileNotFoundError(
            f"Nano root does not exist: {args.root}"
        )

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    suffixes = {
        ".h5",
        ".hdf5",
        ".hdf",
    }

    print("Discovering Nano HDF5 files...", flush=True)

    files = sorted(
        path
        for path in args.root.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in suffixes
        )
    )

    if args.limit is not None:
        files = files[: args.limit]

    print(
        f"Files discovered : {len(files)}",
        flush=True,
    )
    print(
        f"Workers          : {args.workers}",
        flush=True,
    )
    print(
        f"Output           : {args.output}",
        flush=True,
    )

    # --------------------------------------------------------------------------
    # Resume from checkpoint if available.
    # --------------------------------------------------------------------------

    rows: list[dict[str, Any]] = []
    completed_paths: set[str] = set()

    if (
        args.output.exists()
        and not args.overwrite
    ):
        previous = pd.read_csv(
            args.output
        )

        if "file_path" not in previous.columns:
            raise ValueError(
                f"Existing output has no file_path column: {args.output}"
            )

        rows = previous.to_dict(
            orient="records"
        )

        completed_paths = set(
            previous[
                "file_path"
            ]
            .dropna()
            .astype(str)
        )

        print(
            f"Resuming from checkpoint: "
            f"{len(completed_paths)} files already scanned.",
            flush=True,
        )

    pending = [
        path
        for path in files
        if str(path) not in completed_paths
    ]

    print(
        f"Remaining files  : {len(pending)}",
        flush=True,
    )

    if not pending:
        print(
            "Nothing to do. Manifest is already complete.",
            flush=True,
        )
        return

    # spawn is safer than forking an already initialized HDF5 process.
    context = mp.get_context(
        "spawn"
    )

    newly_completed = 0
    worker_failures = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            args.label,
            str(args.root),
            args.priority,
            args.notes,
        ),
    ) as executor:

        futures = {
            executor.submit(
                _scan_one,
                str(path),
            ): path
            for path in pending
        }

        for future in as_completed(
            futures
        ):
            path = futures[
                future
            ]

            try:
                result = future.result()

            except Exception as exc:
                # scan_file itself normally converts HDF5 problems into an
                # error row. This is only for unexpected worker failures.
                worker_failures += 1

                result = {
                    column: None
                    for column in COLUMNS
                }

                result.update(
                    {
                        "recording_label": args.label,
                        "recording_priority": args.priority,
                        "recording_notes": args.notes,
                        "file_path": str(path),
                        "error": (
                            f"WorkerError: "
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }
                )

            rows.append(
                result
            )

            newly_completed += 1

            total_completed = (
                len(completed_paths)
                + newly_completed
            )

            if (
                newly_completed
                % args.progress_every
                == 0
                or newly_completed == len(pending)
            ):
                errors = sum(
                    bool(
                        str(
                            row.get(
                                "error",
                                ""
                            )
                        ).strip()
                        not in {
                            "",
                            "None",
                            "nan",
                        }
                    )
                    for row in rows
                )

                print(
                    f"[{total_completed:5d}/{len(files):5d}] "
                    f"new={newly_completed:5d} "
                    f"errors={errors:4d}",
                    flush=True,
                )

            if (
                newly_completed
                % args.checkpoint_every
                == 0
            ):
                _atomic_write(
                    rows,
                    args.output,
                )

                print(
                    f"Checkpoint saved: "
                    f"{args.output}",
                    flush=True,
                )

    _atomic_write(
        rows,
        args.output,
    )

    final = pd.read_csv(
        args.output
    )

    error_mask = (
        final["error"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    valid_times = (
        pd.to_datetime(
            final["start_time_utc"],
            utc=True,
            errors="coerce",
        ).notna()
        &
        pd.to_datetime(
            final["end_time_utc"],
            utc=True,
            errors="coerce",
        ).notna()
    )

    print()
    print("Nano scan complete")
    print("------------------")
    print(
        f"rows             : {len(final)}"
    )
    print(
        f"valid timed rows : {valid_times.sum()}"
    )
    print(
        f"scan errors      : {error_mask.sum()}"
    )
    print(
        f"worker failures  : {worker_failures}"
    )
    print(
        f"saved            : {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()