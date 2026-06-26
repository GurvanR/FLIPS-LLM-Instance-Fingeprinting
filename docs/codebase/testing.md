# Testing

Project test strategy. Keep this file as a portal — link out to specific test guides (unit,
integration, e2e) as they accumulate.

---

## Test suite

### Smoke test

Runs the classification pipeline end-to-end against the committed in-repo light subset
(`Productions/FLiPS_ICML_light_subset/`) with no data download required.

```bash
make smoke
# or equivalently:
python -m pytest tests/smoke/ -x -q --timeout=120
```

Tests marked with `pytest.mark.requires_data` are skipped when the downloaded Zenodo archives are
absent. The smoke suite must stay green on a fresh clone (zero download).

### Import smoke

Verifies that the package is importable and the analysis stack loads correctly:

```bash
python -c "import audit_llm"
```

This catches broken `__init__.py` re-exports, missing transitive dependencies, and import-time
errors in the analysis stack (numpy, pandas, scikit-learn, xgboost, pyarrow) without running any
inference code. The `generation` extra (vLLM, transformers) is not required for this check.

### Lint

```bash
flake8 src/audit_llm/
```

The flake8 config lives in `setup.cfg` (max line length 120). Lint is **advisory**: the codebase
carries pre-existing style findings inherited from the research repo, so CI runs flake8 as a
non-blocking step rather than a hard gate. Run it before committing to avoid adding new issues.

### SHA-256 manifest verification

```bash
python scripts/fetch_data.py --check-only
```

Classifies every entry in `data/manifest.sha256` as PASS / FAIL / MISSING / EXTERNAL:
- **PASS**: file present and hash matches.
- **FAIL**: file present but hash mismatch — indicates corrupted or stale data.
- **MISSING**: file not yet fetched (benign for optional archives).
- **EXTERNAL**: zero-hash entry (the off-disk `FLiPS_ICML_run`) — skipped.

Exits non-zero only on a hash mismatch (FAIL). Use after `scripts/fetch_data.py` to confirm
download integrity.

---

## Gate scripts

| Script | What it checks |
|--------|---------------|
| `scripts/gates/gate_a.py` (`make gate-a`) | Runs the LLMmap-only out-of-box E3 curve + light-subset smoke; zero download |
| `scripts/gates/gate_golden.py` (`make gate-golden`) | Re-runs `Smoke_light` config and asserts accuracy bands against `golden_refs.json` |
| `scripts/gates/gate_b_parity.py` (`make gate-b-parity`) | Class-parity firewall: enumerator label set == legacy path label set |

---

## Continuous Integration

`.github/workflows/ci.yml` runs on every push and PR to `main`:

1. **Lint (advisory)** — `flake8 src/audit_llm/` (non-blocking; pre-existing research-repo style findings)
2. **Import smoke** — `python -c "import audit_llm; print('audit_llm OK')"`
3. **Light smoke test** — `pytest tests/smoke/ -x -q --timeout=120` (no data download; skips `requires_data`-marked tests)

---

## pytest markers

| Marker | Meaning |
|--------|---------|
| `requires_data` | Test requires a downloaded Zenodo archive; skip when absent |

Markers are declared in `pyproject.toml` under `[tool.pytest.ini_options]`.
