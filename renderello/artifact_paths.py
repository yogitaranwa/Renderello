"""Resolve archived run artifacts without searching unrelated directories."""
from pathlib import Path, PurePosixPath


def run_artifact(run: Path, recorded: str) -> Path | None:
    """Use only a path contained in this exact run. Old machine prefixes are data.

    Accept relative paths or an absolute Windows/POSIX path anchored by the
    exact unique run folder name. No basename-only guessing or parent traversal.
    """
    if not recorded:
        return None
    run = run.resolve()
    parts = PurePosixPath(recorded.replace("\\", "/")).parts
    if ".." in parts:
        return None
    if run.name in parts:
        # Repeated run-name anchors are ambiguous, never pick one silently.
        if parts.count(run.name) != 1:
            return None
        tail = parts[parts.index(run.name) + 1:]
    elif not PurePosixPath(recorded.replace("\\", "/")).is_absolute() and not (len(parts) and ":" in parts[0]):
        tail = parts
    else:
        return None
    if not tail:
        return None
    candidate = run.joinpath(*tail).resolve()
    if run not in candidate.parents or not candidate.is_file():
        return None
    return candidate
