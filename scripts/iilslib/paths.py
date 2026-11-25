from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository/project root inferred from this helper module."""
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path: str | Path, root: Path | None = None) -> Path:
    """Resolve a user-supplied path relative to the project root.

    Absolute paths are preserved. Relative paths are interpreted relative to
    root when supplied, otherwise relative to the project directory.
    """
    path = Path(path)
    if path.is_absolute():
        return path
    base = root if root is not None else project_root()
    return base / path


def resolve_manifest_path(raw_path: str, manifest_path: Path) -> Path:
    """Resolve a path value read from a manifest TSV.

    New manifests use project-root-relative paths. Older experimental
    manifests may contain paths relative to the manifest parent, so keep that
    fallback when it points at an existing file or directory.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path

    root = project_root()
    project_candidate = root / path
    if project_candidate.exists():
        return project_candidate

    manifest_candidate = manifest_path.parent / path
    if manifest_candidate.exists():
        return manifest_candidate

    if has_existing_parent_below(project_candidate, root):
        return project_candidate
    if has_existing_parent_below(manifest_candidate, manifest_path.parent):
        return manifest_candidate

    return project_candidate


def has_existing_parent_below(path: Path, base: Path) -> bool:
    """Return true when a path has an existing parent below base."""
    path = path.resolve()
    base = base.resolve()
    current = path.parent
    while current != current.parent:
        if current == base:
            return False
        if current.exists():
            return True
        current = current.parent
    return False


def project_path_text(path: str | Path) -> str:
    """Return a project-root-relative path string when possible."""
    return emit_path(Path(path), base=project_root(), mode="relative")


def ensure_can_write(path: Path, force: bool) -> None:
    """Raise if path exists and overwrite was not explicitly allowed."""
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite existing file without --force: {path}"
        )


def emit_path(path: Path, base: Path | None = None, mode: str = "as-is") -> str:
    """Return a path string in a stable style for manifests.

    mode:
        as-is: return str(path)
        absolute: return resolved absolute path
        relative: return path relative to base when possible
    """
    if mode == "as-is":
        return str(path)
    if mode == "absolute":
        return str(path.resolve())
    if mode == "relative":
        if base is None:
            base = project_root()
        path_abs = resolve_project_path(path).resolve()
        base_abs = resolve_project_path(base).resolve()
        try:
            return str(path_abs.relative_to(base_abs))
        except ValueError:
            return str(path_abs)
    raise ValueError(f"unsupported path mode: {mode}")


def resolve_list_path(raw_path: str, list_file: Path, root: Path | None = None) -> Path:
    """Resolve a path read from a list file.

    Absolute paths are returned unchanged. Relative paths are first interpreted
    relative to root, when supplied, and otherwise relative to the list file's
    parent directory.
    """
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if root is not None:
        return root / path
    return list_file.parent / path
