#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: CC-BY-4.0
"""Generate SHA-256 manifest lines for a data run directory (stdlib only).

Given a directory (opened READ-ONLY), walk every regular file, compute its
SHA-256, and emit one ``<sha256>  <relative/path>`` line per file, sorted by
path. The output format is exactly that of ``data/manifest.sha256``: two spaces
between the hex digest and the path (the ``sha256sum`` convention), paths using
forward slashes regardless of platform.

The emitted path is ``<prefix>/<path-relative-to-root>`` so a member archive can
be hashed at its real on-disk location while the manifest records the path it
will occupy inside the repository (e.g. ``Productions/FLiPS_ICML_light_subset/...``).

This is the SAME script an external user re-runs after rebuilding an archive
(see ``data/README.md``): point it at the unpacked run directory and paste the
hashes into ``data/manifest.sha256``.

Examples
--------
# Hash the committed light subset, paths prefixed with its in-repo location:
python scripts/gen_manifest.py Productions/FLiPS_ICML_light_subset \\
    --prefix Productions/FLiPS_ICML_light_subset

# Hash a canonical member that lives outside the repo, recording its in-repo path:
python scripts/gen_manifest.py /abs/path/to/LLMmap_ICML_run \\
    --prefix Productions/LLMmap_ICML_run
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Iterator

CHUNK = 1 << 20  # 1 MiB streaming read — never load a whole npy/parquet into RAM.


def sha256_file(path: Path) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under *root*, skipping symlinks."""
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            yield p


def manifest_lines(root: Path, prefix: str) -> list[str]:
    """Return sorted ``<sha256>  <prefix>/<relpath>`` lines for *root*."""
    prefix = prefix.strip("/")
    lines: list[str] = []
    for f in iter_files(root):
        rel = f.relative_to(root).as_posix()
        out_path = f"{prefix}/{rel}" if prefix else rel
        lines.append(f"{sha256_file(f)}  {out_path}")
    lines.sort(key=lambda ln: ln.split("  ", 1)[1])
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="Run directory to hash (read-only).")
    ap.add_argument(
        "--prefix",
        default="",
        help="Path prefix recorded in the manifest (e.g. Productions/<member>). "
        "Defaults to empty (paths relative to root).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write to this file instead of stdout (appends if --append).",
    )
    ap.add_argument("--append", action="store_true", help="Append to --out instead of overwriting.")
    args = ap.parse_args(argv)

    if not args.root.is_dir():
        print(f"error: not a directory: {args.root}", file=sys.stderr)
        return 1

    lines = manifest_lines(args.root, args.prefix)
    body = "\n".join(lines) + ("\n" if lines else "")

    if args.out is None:
        sys.stdout.write(body)
    else:
        mode = "a" if args.append else "w"
        with open(args.out, mode) as f:
            f.write(body)
        print(f"wrote {len(lines)} hashes -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
