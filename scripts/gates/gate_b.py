#!/usr/bin/env python3
"""Gate-B — class-parity firewall (part a) + coarse-banded accuracy sanity (part b).

Combined gate wired by the Makefile as ``make gate-b``. Part (a) alone is
``make gate-b-parity`` (``scripts/gates/gate_b_parity.py``).

  PART (a) — DATA-FREE FIREWALL (always runs)
    Delegates to ``gate_b_parity.check_parity()``: the enumerator label set MUST
    exactly equal the legacy path's label set, computed from config alone. A diff
    here FAILS the whole gate (this is the firewall the legacy deletion gates on).

  PART (b) — COARSE-BANDED ACCURACY SANITY (separate, explicitly approximate)
    Read-only; NEVER recomputes. Two facets:

    * In-repo LLMmap anchor sanity (always, data-free): the released LLMmap
      ``results.json`` nested ``ds_wise/<bs>/no_token_pairs/llmmap_clf/accuracy_mean``
      is read both raw (key ``ds_wise``, treated as frozen) AND via
      ``load_train_size_dict`` (in-memory key ``tp_wise`` — the normalisation
      ``checkpoint_utils.py`` applies on load). Both must expose the same values;
      bs6≈0.3337 / bs4≈0.3231 within a COARSE band (±5 pp); all 8 batch sizes present
      and in a plausible 205-class LLMmap range — emphatically NOT a top-level ~96%
      (the headline 96% is the FLiPS closed-set number, never this LLMmap key).
      This is the read-correctness + ds_wise→tp_wise normalisation firewall on the
      shipped anchor; a wrong read FAILS the gate.

    * Headline accuracy (DATA-GATED): if ``Productions/<RUN_NAME>/run_config.{json,pickle}``
      (the flattened uploaded ``FLiPS_ICML_run``) is present, the headline run's
      ``results.json`` accuracy_mean is band-checked. When ABSENT (the local build
      machine — the headline run is the one off-disk exception) this prints a LOUD
      SKIP-not-pass message and the gate NEVER reports PASS. The PASS path is
      exercised by the operator during the manual headline-release procedure.

  Why the band is WIDE (≥ ±5 pp mix_tp / ±2 pp tp_wise; any "±1e-6" is dropped):
    recompute variance is real — ``StratifiedShuffleSplit n_splits=2`` +
    ``force_class_size: 500`` (the headline reproduction configs pin it; the smoke uses
    the ``"auto"`` default) AND the UNSEEDED ``random.sample`` in ``token_pair_mixing.py``
    (~±3.4 pp run-to-run on mix_tp, same env/seed — golden-repro finding). Exact
    regression is gate-golden's job, not this sanity.

Final stdout token ``GATE-B-RESULT: PASS | SKIP | FAIL`` so downstream tooling can grep the
outcome. PASS/SKIP exit 0 (SKIP matches the Makefile data-gated SKIP-loud
convention; its message states plainly it did NOT pass); FAIL exits 1.

Needs ``audit_llm`` importable — run inside the poetry ``.venv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))  # audit_llm import fallback (non-editable install)

PRODUCTIONS = REPO_ROOT / "Productions"
LLMMAP_RESULTS = (
    REPO_ROOT
    / "XP_configs"
    / "e3_flips_vs_llmmap"
    / "llmmap_if_data"
    / "ClosedSet"
    / "train_size_checkpoints"
    / "all"
    / "40"
    / "results.json"
)

# Coarse bands (absolute, fraction units) — deliberately wide; see module docstring.
BAND_LLMMAP = 0.05               # ±5 pp around the documented LLMmap anchor
ANCHOR_LLMMAP = {6: 0.3337, 4: 0.3231}   # documented examples
PLAUSIBLE_LLMMAP = (0.15, 0.55)  # a 205-class LLMmap closed-set accuracy lands here
PLAUSIBLE_FLIPS = (0.80, 1.0)    # the FLiPS headline closed/open-set (90-96%) lands here
NOT_PASS_HI = (0.90, 1.0)        # values an LLMmap key must never reach (the ~96% trap)


def _load_parity_module():
    """Load gate_b_parity.py by file path (the sibling-import pattern gate_golden uses)."""
    src = REPO_ROOT / "scripts" / "gates" / "gate_b_parity.py"
    spec = importlib.util.spec_from_file_location("gate_b_parity", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _bs_accuracy_map(wise_dict: dict) -> dict[int, float]:
    """Extract ``{batch_size:int -> accuracy_mean}`` from a ds_wise/tp_wise dict."""
    out: dict[int, float] = {}
    for bs, sub in (wise_dict or {}).items():
        try:
            out[int(bs)] = sub["no_token_pairs"]["llmmap_clf"]["accuracy_mean"]
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _loader_bs_accuracy() -> dict[int, float]:
    """Read the in-repo LLMmap results via load_train_size_dict (in-memory tp_wise)."""
    from audit_llm.xp_tools.checkpoint_utils import load_train_size_dict

    checkpoint_dir = LLMMAP_RESULTS.parents[2]  # .../train_size_checkpoints
    train_size_dict, _ = load_train_size_dict(checkpoint_dir, "LLMmap-IF")
    # train_size_dict = {40: {"all": {"tp_wise": {<bs>: {...}}}}}
    ts = next(iter(train_size_dict.values()))
    item = next(iter(ts.values()))
    return _bs_accuracy_map(item.get("tp_wise", {}))


def check_in_repo_anchor() -> tuple[bool, str]:
    """In-repo LLMmap anchor sanity — read-correctness + ds_wise→tp_wise normalisation."""
    if not LLMMAP_RESULTS.exists():
        return False, f"in-repo LLMmap results.json missing at {LLMMAP_RESULTS.relative_to(REPO_ROOT)}"

    raw = json.loads(LLMMAP_RESULTS.read_text())
    if "ds_wise" not in raw:
        return False, "raw results.json has no FROZEN top key 'ds_wise' (ds_wise invariant broken)"
    raw_acc = _bs_accuracy_map(raw["ds_wise"])

    try:
        loader_acc = _loader_bs_accuracy()
    except Exception as exc:  # noqa: BLE001 — surface any loader breakage as a gate failure
        return False, f"load_train_size_dict raised reading the in-repo cache: {exc!r}"

    if not raw_acc:
        return False, "no <bs>/no_token_pairs/llmmap_clf/accuracy_mean found under ds_wise"
    if set(raw_acc) != set(loader_acc):
        return False, (f"raw ds_wise bs {sorted(raw_acc)} != loader tp_wise bs {sorted(loader_acc)} "
                       "(ds_wise→tp_wise normalisation lost a batch size)")
    drift = {bs: (raw_acc[bs], loader_acc[bs]) for bs in raw_acc if abs(raw_acc[bs] - loader_acc[bs]) > 1e-9}
    if drift:
        return False, f"raw ds_wise vs loader tp_wise accuracy mismatch (normalisation not value-preserving): {drift}"

    # All 8 batch sizes present, each in the plausible LLMmap range, none in the ~96% trap.
    if len(raw_acc) != 8:
        return False, f"expected 8 batch sizes (1..8), got {sorted(raw_acc)}"
    bad_range = {bs: a for bs, a in raw_acc.items() if not (PLAUSIBLE_LLMMAP[0] <= a <= PLAUSIBLE_LLMMAP[1])}
    if bad_range:
        return False, f"LLMmap accuracy outside plausible {PLAUSIBLE_LLMMAP}: {bad_range}"
    trap = {bs: a for bs, a in raw_acc.items() if NOT_PASS_HI[0] <= a <= NOT_PASS_HI[1]}
    if trap:
        return False, f"LLMmap key reads ~96% {trap} — wrong key (anchor is the nested ~0.33, not a top-level 96%)"

    # Documented anchor examples within the coarse band.
    for bs, expected in ANCHOR_LLMMAP.items():
        got = raw_acc.get(bs)
        if got is None or abs(got - expected) > BAND_LLMMAP:
            return False, f"bs={bs} accuracy_mean={got} not within ±{BAND_LLMMAP} of documented {expected}"

    return True, (f"8/8 batch sizes read ~0.33 (bs6={raw_acc[6]:.4f}, bs4={raw_acc[4]:.4f}) "
                  "via raw ds_wise AND loader tp_wise; not a top-level 96%")


def check_headline_accuracy(run_name: str) -> tuple[str, str]:
    """Headline accuracy band-check (DATA-GATED). Returns (status, detail).

    status ∈ {"PASS", "SKIP", "FAIL"}. Locally the headline run is absent → SKIP.
    The PASS path is exercised by the operator during the manual headline-release procedure.
    """
    run_dir = PRODUCTIONS / run_name
    has_headline = (run_dir / "run_config.json").exists() or (run_dir / "run_config.pickle").exists()
    if not has_headline:
        return "SKIP", (f"Productions/{run_name}/run_config.{{json,pickle}} absent — the headline "
                        "FLiPS_ICML_run is off-disk on this build machine. Headline accuracy "
                        "NOT VERIFIED (this is NOT a pass). Run `make fetch-headline` first; "
                        "the headline tier is verified during the manual headline-release procedure.")

    results = sorted(run_dir.rglob("results.json"))
    if not results:
        return "FAIL", f"headline run present but no results.json under Productions/{run_name}/"

    checked = 0
    for rj in results:
        try:
            data = json.loads(rj.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            return "FAIL", f"{rj.relative_to(REPO_ROOT)}: unreadable ({exc!r})"
        wise = data.get("ds_wise") or data.get("tp_wise") or {}
        for bs, acc in _bs_accuracy_map(wise).items():
            in_llmmap = PLAUSIBLE_LLMMAP[0] <= acc <= PLAUSIBLE_LLMMAP[1]
            in_flips = PLAUSIBLE_FLIPS[0] <= acc <= PLAUSIBLE_FLIPS[1]
            if not (in_llmmap or in_flips):
                return "FAIL", (f"{rj.relative_to(REPO_ROOT)}: bs={bs} accuracy_mean={acc:.4f} "
                                f"outside both coarse bands {PLAUSIBLE_LLMMAP} / {PLAUSIBLE_FLIPS}")
            checked += 1
    return "PASS", f"{checked} headline accuracy_mean values within the coarse fingerprinting band"


def main() -> int:
    ap = argparse.ArgumentParser(description="gate-B: data-free parity firewall + accuracy sanity.")
    ap.add_argument("--run-name", default="FLiPS_ICML_run",
                    help="Productions/<run> holding the flattened headline run_config (default: FLiPS_ICML_run).")
    args = ap.parse_args()

    parity = _load_parity_module()

    print("=== gate-B : class-parity firewall (a) + coarse accuracy sanity (b) ===\n")

    # PART (a) — always; a drift here is fatal.
    print("--- part (a) data-free class-parity firewall ---")
    parity_ok, parity_detail = parity.check_parity()
    print(f"  [{'PASS' if parity_ok else 'FAIL'}] {parity_detail}\n")

    # PART (b) — in-repo anchor sanity (always) + headline accuracy (data-gated).
    print("--- part (b) coarse-banded accuracy sanity ---")
    anchor_ok, anchor_detail = check_in_repo_anchor()
    print(f"  [{'PASS' if anchor_ok else 'FAIL'}] in-repo LLMmap anchor: {anchor_detail}")
    headline_status, headline_detail = check_headline_accuracy(args.run_name)
    tag = {"PASS": "PASS", "SKIP": "SKIP", "FAIL": "FAIL"}[headline_status]
    print(f"  [{tag}] headline accuracy: {headline_detail}\n")

    # Combined outcome.
    if not parity_ok or not anchor_ok or headline_status == "FAIL":
        result = "FAIL"
    elif headline_status == "SKIP":
        result = "SKIP"
    else:
        result = "PASS"

    print("=" * 70)
    if result == "PASS":
        print("gate-B: PASS — parity firewall held AND headline accuracy within band.")
    elif result == "SKIP":
        print("gate-B: SKIP-WITHOUT-PASS — parity firewall held (part a) and the in-repo LLMmap")
        print("        anchor reads correctly, but the headline FLiPS_ICML_run is absent, so the")
        print("        headline accuracy tier was NOT verified. This is the expected local outcome;")
        print("        it does NOT count as a pass. Verified during the manual headline-release procedure.")
    else:
        print("gate-B: FAIL — see the [FAIL] line(s) above.")
    print(f"GATE-B-RESULT: {result}")
    print("=" * 70)

    # PASS/SKIP exit 0 (SKIP-loud convention); FAIL exits 1.
    return 1 if result == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
