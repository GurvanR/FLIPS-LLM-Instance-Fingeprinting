<!--
SPDX-FileCopyrightText: 2026 Pôle d'Expertise de la Régulation Numérique <contact@peren.gouv.fr>
SPDX-License-Identifier: CC-BY-4.0
-->
# `data/` — released data members, manifest, and fetch wiring

This directory documents the FLiPS released data: what ships **in the repo**, what
is **fetched from an external archive**, and how the two are verified. The data
license (and the per-model output constraints layered on top) is in
[`LICENSE`](LICENSE).

## Files

- [`manifest.sha256`](manifest.sha256) — SHA-256 of every released file, one
  `<sha256>  <relative/path>` line per file (sha256sum convention). The header
  enumerates the heterogeneous sources by name + role + size. Entries whose hash is
  **64 zeros** are off-disk / externally-hosted and are skipped as external by the
  verifier (never PASS/FAIL).
- [`LICENSE`](LICENSE) — CC-BY-4.0 for our data/features, plus the Cohere (CC-BY-NC)
  and Gemma (ToU) per-model output constraints, and the no-weight-redistribution note.

## In-repo vs external

| Member | Where | Role |
|---|---|---|
| `Productions/FLiPS_ICML_light_subset/` | **committed in this repo** (~92 MB) | zero-download CPU smoke fixture (5 token pairs); backs `make smoke` / `make gate-golden` |
| `Productions/Intersection_vocab/` | **committed in this repo** | tokenizer-vocab intersection (import-time dep) |
| `Productions/LLMmap_ICML_run/` | Zenodo — [record 20733999](https://zenodo.org/records/20733999) (~5.3 GB, 600 files) | 205-class LLMmap traces (LLMmap E3) |
| `Productions/FLiPS_ICML_run/` | Zenodo — **same record** (81 files: `Analysis/` + `run_config.json`, no `Experiments/`) | 96%/90% E1/E2 + FLIPS E3 arms — gates the headline tier |

Both external members ship in **one combined archive** on that record
(`flips_release_data.tar.gz`, open access). `fetch_data.py` downloads it once and
places each member: `LLMmap_ICML_run` keeps its `Productions/…` prefix, `FLiPS_ICML_run`
is **flattened** into `Productions/FLiPS_ICML_run/`. The regenerable `Experiments/`
outputs are deliberately **not** archived — they are rebuilt by `make repro-*`.

The committed light subset is **flattened**: `run_config.{json,pickle}` and the
`Analysis/` / `Experiments/` subdirs sit directly under
`Productions/FLiPS_ICML_light_subset/` (no `merged_sub_runs/` level), which is the
run-path convention `AuditionsAnalysis(run_path)` expects. It carries
`golden_refs.json` (the out-of-box golden regression reference: class label set,
per-cache md5s, and the smoke accuracy band).

## The headline data (published on Zenodo)

The **headline numbers** (96% closed-set / 90% open-set + the FLIPS E3 arms) require
the `FLiPS_ICML_run` export from the HPC cluster. It is too large to commit, but it is
now **published on Zenodo** (in the combined archive above) with **real SHA-256 hashes**
in `manifest.sha256` — no longer a 64-zero placeholder. On a fresh checkout it is still
absent from disk, so the data-gated `make repro-*` targets SKIP loudly until you
**fetch** it (`fetch_data.py --with-headline`); only then is the headline tier
reproducible. Publishing / refreshing this record is a manual, by-hand operator step.

## Fetching + verifying (`scripts/fetch_data.py`)

`scripts/fetch_data.py` (stdlib only) downloads the released archive(s), unpacks them
into the `Productions/` layout, and verifies **every** unpacked file's SHA-256 against
`manifest.sha256` — aborting and rolling back on any mismatch.

The data URL is resolved by priority: `--url` > `$FLIPS_DATA_URL` > the
`ZENODO_RECORD_URL` constant in the script (now wired to
`https://zenodo.org/records/20733999`; before upload it was a `PLACEHOLDER_DOI`, against
which a real fetch is refused). Use the `…/records/<id>` form — not a DOI or a tokened
preview link — since the resolver reads the record's files via the public Zenodo API.

| Flag | Effect |
|---|---|
| `--check-only` | Verify present members against the manifest; **zero network**. Off-disk / placeholder (64-zero) entries are skipped as external; not-yet-fetched members are reported as missing (not a failure). Exit 1 only on a hash mismatch. |
| `--dry-run` | Print the fetch plan (records, members, target paths); no download, no writes. |
| `--with-headline` | Also fetch the off-disk headline `FLiPS_ICML_run`, unpacked **flattened** into `Productions/FLiPS_ICML_run/`. |
| `--with-generations` | Also fetch the optional raw-generations / D1 re-parse record. |
| `--manifest`, `--dest`, `--url` | Manifest path, repo root to unpack into, source URL override. |

- **Verify present members offline** (no network):
  `python scripts/fetch_data.py --check-only --manifest data/manifest.sha256`.
  This validates the committed light subset + `Intersection_vocab` with zero downloads.
- **Rebuild a member's hashes** after (re)building its archive:
  `python scripts/gen_manifest.py <member-dir> --prefix Productions/<member>`,
  then paste the emitted lines into `manifest.sha256` and commit.

### Upload-then-wire (publishing / re-versioning the external archives)

**Status:** done for the current release — both members are published in one combined
archive on [record 20733999](https://zenodo.org/records/20733999) and the URL is wired
in `scripts/fetch_data.py`. The steps below are the reusable runbook for re-versioning.

1. Build the archive(s) for the external members. The combined archive keeps each
   member under its `Productions/<member>/` prefix (LLMmap is placed verbatim;
   `FLiPS_ICML_run` is flattened on download), and **excludes** the regenerable
   `FLiPS_ICML_run/Experiments/`.
2. Upload to Zenodo and **publish** (open access); copy the record id from the record
   URL `https://zenodo.org/records/<id>`.
3. Wire the URL — **either** edit `ZENODO_RECORD_URL` in `scripts/fetch_data.py` to
   `https://zenodo.org/records/<id>` and commit, **or** set
   `export FLIPS_DATA_URL=https://zenodo.org/records/<id>` per run. Use the
   `…/records/<id>` form (not a DOI or tokened preview link).
4. `python scripts/gen_manifest.py Productions/<run> --prefix Productions/<run>` for each
   uploaded member and update `manifest.sha256` (replacing any 64-zero placeholder with
   real hashes); commit `data/manifest.sha256`. Sanity-check that the manifest's
   archived-member rows match the archive exactly — neither phantom rows nor missing files.
5. Fetch + verify: `python scripts/fetch_data.py` (core tier) and
   `python scripts/fetch_data.py --with-headline` (headline tier).

On a **fresh checkout** the headline tier is still SKIPPED — `FLiPS_ICML_run` is not
committed, so the data-gated `make repro-*` targets skip loudly until you run
`fetch_data.py --with-headline` to pull and verify it locally.

## No weight redistribution

Only derived bit-sequence features / classification artefacts are released here.
Model weights are obtained from their original sources under their own licenses.
