#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
#
# SPDX-License-Identifier: CC-BY-4.0
"""Fetch + verify the FLiPS released data archives (stdlib only).

Downloads the released data archive(s) from a Zenodo record (or a direct archive
URL), unpacks them into the repository's ``Productions/`` layout, and verifies
**every** unpacked file's SHA-256 against ``data/manifest.sha256``. Hash mismatch
aborts and rolls back the offending member; already-present + valid files are
skipped (idempotent re-run).

URL resolution (priority): ``--url`` > ``$FLIPS_DATA_URL`` > the module constant
``ZENODO_RECORD_URL`` (wired to the published record). A download against an unset
``PLACEHOLDER_DOI`` is refused; ``--dry-run`` / ``--check-only`` never touch the network.

Records (heterogeneous sources, each separately fetchable):

* ``core``        (out-of-box, always): ``LLMmap_ICML_run`` (backs the LLMmap E3
  curve). Lands at its manifest path.
* ``headline``    (data-gated, ``--with-headline``): the OFF-DISK ``FLiPS_ICML_run``
  headline export. Unpacked **FLATTENED** into ``Productions/FLiPS_ICML_run/`` —
  ``run_config.*`` + ``Analysis/``/``Experiments/`` directly under the run path, no
  ``merged_sub_runs/`` level (the one run-path convention shared with the
  light subset, so ``AuditionsAnalysis`` finds ``run_config`` under ``run_path``).
* ``generations`` (Mode-B, ``--with-generations``): optional raw-generations / D1
  re-parse record (no manifest rows until the user uploads it).

The in-repo ``FLiPS_ICML_light_subset/`` and ``Intersection_vocab/`` are committed,
not fetched — but ``--check-only`` still verifies them since they are present.

Examples
--------
# Verify the committed subset offline (zero network):
python scripts/fetch_data.py --check-only --manifest data/manifest.sha256

# Show what a fetch would do, including the headline tier:
python scripts/fetch_data.py --dry-run --with-headline

# Real fetch from the wired record (override with env or --url):
python scripts/fetch_data.py --with-headline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

# --- Constants ---------------------------------------------------------------

# Published Zenodo record holding flips_release_data.tar.gz (LLMmap_ICML_run core +
# FLiPS_ICML_run headline, both members in one archive). Override per-run with
# --url / $FLIPS_DATA_URL. Must stay the …/records/<id> form — the resolver parses
# that shape via the public API; a DOI or tokened preview URL will not work here.
ZENODO_RECORD_URL = "https://zenodo.org/records/20733999"

PLACEHOLDER_HASH = "0" * 64  # off-disk / externally-hosted manifest entries.
PRODUCTIONS = "Productions"  # top-level data dir these records unpack into.
CHUNK = 1 << 20  # 1 MiB streaming reads — never load a whole archive/file into RAM.


@dataclass(frozen=True)
class Record:
    """A separately-fetchable archive grouping one or more top-level members."""

    name: str
    tier: str
    members: tuple[str, ...]
    flatten: bool = False  # strip an intermediate run-dir level (e.g. merged_sub_runs/).
    gated_by: str | None = None  # CLI flag that must be set to select this record.


RECORDS: tuple[Record, ...] = (
    Record(
        name="core",
        tier="out-of-box",
        members=("LLMmap_ICML_run",),
    ),
    Record(
        name="headline",
        tier="data-gated",
        members=("FLiPS_ICML_run",),
        flatten=True,
        gated_by="--with-headline",
    ),
    Record(
        name="generations",
        tier="mode-b",
        members=(),  # raw generations / D1 re-parse — manifest rows filled when uploaded.
        gated_by="--with-generations",
    ),
)


# --- Manifest ----------------------------------------------------------------


@dataclass
class Entry:
    sha256: str
    relpath: str  # repo-root-relative, forward slashes (manifest convention).

    @property
    def is_external(self) -> bool:
        return self.sha256 == PLACEHOLDER_HASH

    @property
    def member(self) -> str:
        parts = self.relpath.split("/")
        if parts and parts[0] == PRODUCTIONS and len(parts) > 1:
            return parts[1]
        return parts[0] if parts else self.relpath


def parse_manifest(path: Path) -> list[Entry]:
    """Parse a sha256sum-style manifest, skipping ``#`` comments and blanks."""
    entries: list[Entry] = []
    for raw in path.read_text().splitlines():
        line = raw.rstrip("\n")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # Two-space separator per sha256sum convention; tolerate extra spaces in paths.
        sha, sep, rel = line.partition("  ")
        if not sep:
            raise ValueError(f"malformed manifest line (no '  ' separator): {line!r}")
        entries.append(Entry(sha256=sha.strip(), relpath=rel.strip()))
    return entries


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


# --- check-only --------------------------------------------------------------


def check_only(entries: list[Entry], dest: Path) -> int:
    """Verify every present manifest member against its hash. Zero network.

    Classification per entry: EXTERNAL (placeholder hash, skipped), MISSING (not on
    disk — not yet fetched), PASS (present + hash matches), FAIL (present + differs).
    Returns process exit code: 1 iff any FAIL, else 0.
    """
    counts = {"pass": 0, "fail": 0, "missing": 0, "external": 0}
    per_member: dict[str, dict[str, int]] = {}
    failures: list[str] = []

    for e in entries:
        bucket = per_member.setdefault(e.member, {"pass": 0, "fail": 0, "missing": 0, "external": 0})
        if e.is_external:
            counts["external"] += 1
            bucket["external"] += 1
            continue
        target = dest / e.relpath
        if not target.is_file():
            counts["missing"] += 1
            bucket["missing"] += 1
            continue
        if sha256_file(target) == e.sha256:
            counts["pass"] += 1
            bucket["pass"] += 1
        else:
            counts["fail"] += 1
            bucket["fail"] += 1
            failures.append(e.relpath)

    print("check-only — verifying present members against data/manifest.sha256 (no network)\n")
    for member in sorted(per_member):
        b = per_member[member]
        flags = []
        if b["pass"]:
            flags.append(f"{b['pass']} pass")
        if b["fail"]:
            flags.append(f"{b['fail']} FAIL")
        if b["missing"]:
            flags.append(f"{b['missing']} missing")
        if b["external"]:
            flags.append(f"{b['external']} external")
        status = "OK" if not b["fail"] else "FAILED"
        if b["pass"] == 0 and b["fail"] == 0 and b["external"] == 0:
            status = "not fetched"
        print(f"  [{status:>11}] {member:<28} {', '.join(flags)}")

    if failures:
        print("\nHASH MISMATCHES (data is corrupt or stale):")
        for rel in failures:
            print(f"  FAIL  {rel}")

    print(
        f"\nTotals: {counts['pass']} pass, {counts['fail']} fail, "
        f"{counts['missing']} missing (not fetched), {counts['external']} external (skipped)."
    )
    if counts["fail"]:
        print("RESULT: FAIL — at least one present file does not match its manifest hash.")
        return 1
    print("RESULT: OK — every present file matches its manifest hash.")
    return 0


# --- Record selection / planning --------------------------------------------


def selected_records(with_headline: bool, with_generations: bool) -> list[Record]:
    flags = {"--with-headline": with_headline, "--with-generations": with_generations}
    out: list[Record] = []
    for rec in RECORDS:
        if rec.gated_by is None or flags.get(rec.gated_by, False):
            out.append(rec)
    return out


def entries_for_members(entries: list[Entry], members: Iterable[str]) -> list[Entry]:
    members = set(members)
    return [e for e in entries if e.member in members and not e.is_external]


def dry_run(entries: list[Entry], records: list[Record], url: str, dest: Path) -> int:
    print("dry-run — fetch plan (no download, no writes)\n")
    resolved_placeholder = "PLACEHOLDER_DOI" in url
    print(f"  source URL : {url}")
    if resolved_placeholder:
        print("               ^ PLACEHOLDER — a real fetch will refuse. Set --url, $FLIPS_DATA_URL,")
        print("                 or edit ZENODO_RECORD_URL after uploading to Zenodo (see data/README.md).")
    print(f"  manifest   : {len(entries)} entries\n")

    for rec in records:
        print(f"  record '{rec.name}' [{rec.tier}]"
              + (f"  (selected by {rec.gated_by})" if rec.gated_by else "  (always)"))
        if not rec.members:
            print("      (no manifest members yet — filled when the user uploads this record)")
            continue
        for member in rec.members:
            member_entries = entries_for_members(entries, [member])
            if not member_entries:
                print(f"      - {member}: (no manifest rows; off-disk placeholder until uploaded)")
                continue
            total_bytes = 0
            present = 0
            for e in member_entries:
                target = dest / e.relpath
                if target.is_file():
                    present += 1
                    total_bytes += target.stat().st_size
            layout = " FLATTENED into Productions/%s/" % member if rec.flatten else ""
            print(f"      - {member}: {len(member_entries)} files, "
                  f"{present} already present.{layout}")
    print("\nWould verify each unpacked file's SHA-256 against the manifest and roll back on mismatch.")
    return 0


# --- Download + unpack -------------------------------------------------------


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract(archive: Path, into: Path) -> None:
    """Extract a tar/zip archive, refusing any member that escapes *into* (zip-slip)."""
    into.mkdir(parents=True, exist_ok=True)
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for m in tf.getmembers():
                if not _is_within(into, into / m.name):
                    raise RuntimeError(f"refusing path-traversal archive member: {m.name!r}")
            tf.extractall(into)
    elif zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for n in zf.namelist():
                if not _is_within(into, into / n):
                    raise RuntimeError(f"refusing path-traversal archive member: {n!r}")
            zf.extractall(into)
    else:
        raise RuntimeError(f"unrecognised archive format (not tar/zip): {archive}")


def _zenodo_file_urls(record_url: str) -> list[str]:
    """Resolve a Zenodo ``…/records/<id>`` web URL to its archive download URLs.

    Uses the public records API (``…/api/records/<id>``); each ``files[]`` entry
    carries ``links.self`` (a ``…/content`` direct URL). A non-Zenodo URL is
    returned verbatim (treated as a single direct archive URL).
    """
    if "zenodo.org/records/" not in record_url:
        return [record_url]
    rid = record_url.rstrip("/").rsplit("/", 1)[-1]
    api = f"https://zenodo.org/api/records/{rid}"
    with urllib.request.urlopen(api) as resp:  # noqa: S310 — https Zenodo API only.
        meta = json.load(resp)
    urls = [f["links"]["self"] for f in meta.get("files", []) if "links" in f]
    if not urls:
        raise RuntimeError(f"Zenodo record {rid} exposes no downloadable files")
    return urls


def _download(url: str, dest_file: Path) -> None:
    with urllib.request.urlopen(url) as resp, open(dest_file, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out, CHUNK)


def _find_run_root(tree: Path) -> Path:
    """Locate the directory holding ``run_config.*`` within an extracted tree (flatten)."""
    for cfg in sorted(tree.rglob("run_config.*")):
        return cfg.parent
    raise RuntimeError(f"flattened record has no run_config.* under {tree}")


def _place_and_verify(
    src_root: Path,
    record: Record,
    by_relpath: dict[str, Entry],
    dest: Path,
) -> None:
    """Copy a record's members from an extracted tree into the repo, verifying each file.

    Raises on any hash mismatch (caller rolls back). Idempotent: a destination file
    already present + hash-valid is left untouched.
    """
    for member in record.members:
        member_prefix = f"{PRODUCTIONS}/{member}/"
        member_entries = [e for rel, e in by_relpath.items() if rel.startswith(member_prefix)]
        if not member_entries:
            continue  # off-disk placeholder member (e.g. not yet uploaded) — nothing to place.

        if record.flatten:
            run_root = _find_run_root(src_root)  # strips merged_sub_runs/ etc.
        for e in member_entries:
            target = dest / e.relpath
            if target.is_file() and sha256_file(target) == e.sha256:
                print(f"      skip (present+valid): {e.relpath}")
                continue
            rel_in_member = e.relpath[len(member_prefix):]
            source = (run_root / rel_in_member) if record.flatten else (src_root / e.relpath)
            if not source.is_file():
                raise RuntimeError(f"archive missing expected member file: {e.relpath}")
            digest = sha256_file(source)
            if digest != e.sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch for {e.relpath}: expected {e.sha256}, got {digest}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"      ok: {e.relpath}")


def fetch(entries: list[Entry], records: list[Record], url: str, dest: Path) -> int:
    if "PLACEHOLDER_DOI" in url:
        print(
            "error: the data URL is the PLACEHOLDER_DOI. Refusing to download.\n"
            "       Upload the archive(s) to Zenodo and set --url / $FLIPS_DATA_URL, or edit\n"
            "       ZENODO_RECORD_URL in scripts/fetch_data.py. See data/README.md.",
            file=sys.stderr,
        )
        return 2

    by_relpath = {e.relpath: e for e in entries if not e.is_external}
    archive_urls = _zenodo_file_urls(url)
    print(f"resolved {len(archive_urls)} archive URL(s) from {url}")

    tmp = Path(tempfile.mkdtemp(prefix="flips_fetch_"))
    placed_members: list[str] = []
    try:
        for rec in records:
            if not entries_for_members(entries, rec.members):
                print(f"record '{rec.name}': no manifest rows to fetch — skipping.")
                continue
            print(f"record '{rec.name}' [{rec.tier}] -> members {', '.join(rec.members)}")
            for au in archive_urls:
                archive = tmp / Path(au.split("?", 1)[0]).name
                if not archive.name or archive.name == "content":
                    archive = tmp / f"{rec.name}_archive"
                print(f"  downloading {au}")
                _download(au, archive)
                extract_dir = tmp / f"{rec.name}_unpacked"
                _safe_extract(archive, extract_dir)
                _place_and_verify(extract_dir, rec, by_relpath, dest)
                archive.unlink(missing_ok=True)
                shutil.rmtree(extract_dir, ignore_errors=True)
            placed_members.extend(rec.members)
    except Exception as exc:  # noqa: BLE001 — abort + roll back any partial member.
        print(f"\nFETCH ABORTED: {exc}", file=sys.stderr)
        for member in placed_members:
            partial = dest / PRODUCTIONS / member
            if partial.is_dir():
                shutil.rmtree(partial, ignore_errors=True)
                print(f"  rolled back {partial}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\nfetch complete — all unpacked files verified against the manifest.")
    return 0


# --- CLI ---------------------------------------------------------------------


def resolve_url(cli_url: str | None) -> str:
    if cli_url:
        return cli_url
    env = os.environ.get("FLIPS_DATA_URL")
    if env:
        return env
    return ZENODO_RECORD_URL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default=None, help="Zenodo record URL or direct archive URL (overrides env/constant).")
    ap.add_argument("--manifest", type=Path, default=Path("data/manifest.sha256"),
                    help="SHA-256 manifest to verify against (default: data/manifest.sha256).")
    ap.add_argument("--dest", type=Path, default=Path("."),
                    help="Repository root to unpack/verify into (default: current dir).")
    ap.add_argument("--dry-run", action="store_true", help="Print the fetch plan; no download, no writes.")
    ap.add_argument("--check-only", action="store_true",
                    help="Verify present members against the manifest; no network.")
    ap.add_argument("--with-headline", action="store_true",
                    help="Also fetch the OFF-DISK headline FLiPS_ICML_run (flattened).")
    ap.add_argument("--with-generations", action="store_true",
                    help="Also fetch the optional raw-generations / D1 re-parse record.")
    args = ap.parse_args(argv)

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    entries = parse_manifest(args.manifest)

    if args.check_only:
        return check_only(entries, args.dest)

    records = selected_records(args.with_headline, args.with_generations)
    url = resolve_url(args.url)

    if args.dry_run:
        return dry_run(entries, records, url, args.dest)

    return fetch(entries, records, url, args.dest)


if __name__ == "__main__":
    raise SystemExit(main())
