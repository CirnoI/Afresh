from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from .paths import ensure_can_write, resolve_list_path


def read_tsv(path: Path, required: Iterable[str] = ()) -> list[dict[str, str]]:
    """Read a tab-delimited table into dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"TSV file does not exist: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = reader.fieldnames or []
        missing = set(required) - set(fieldnames)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {', '.join(sorted(missing))}"
            )
        return [dict(row) for row in reader]


def write_tsv(
    path: Path,
    fieldnames: list[str],
    rows: Iterable[dict[str, object]],
    force: bool = False,
) -> None:
    """Write dictionaries as a tab-delimited table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_can_write(path, force)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def read_list_file(
    path: Path,
    *,
    root: Path | None = None,
    skip_comments: bool = True,
    resolve_paths: bool = False,
) -> list[str]:
    """Read a one-entry-per-line text list.

    Blank lines are ignored. Lines beginning with '#' are ignored when
    skip_comments is true. If resolve_paths is true, entries are resolved using
    resolve_list_path() and returned as strings.
    """
    if not path.exists():
        raise FileNotFoundError(f"list file does not exist: {path}")
    values: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if skip_comments and line.startswith("#"):
                continue
            if resolve_paths:
                values.append(str(resolve_list_path(line, path, root=root)))
            else:
                values.append(line)
    return values


def write_list_file(path: Path, values: Iterable[str], force: bool = False) -> None:
    """Write a one-entry-per-line text list."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_can_write(path, force)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(str(value))
            handle.write("\n")

