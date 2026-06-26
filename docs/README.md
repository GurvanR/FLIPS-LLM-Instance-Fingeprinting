# Documentation

User/admin-facing docs (installation, running, troubleshooting) live here at the top level. Internal codebase docs live under `codebase/`.

## Reproduction & provenance

- [`reproduction.md`](reproduction.md) — step-by-step Mode A (3 tiers) and Mode B reproduction.
  - [`reproduction/mode-a.md`](reproduction/mode-a.md) — no-GPU analysis path, full command/figure tables.
  - [`reproduction/mode-b.md`](reproduction/mode-b.md) — GPU from-scratch inference path.
  - [`reproduction/custom-scenario.md`](reproduction/custom-scenario.md) — run your own scenario
    (your own model × variation set) end to end, with a runnable example.
- [`provenance.md`](provenance.md) — the 34→205→237 chain, the 25 LLMs + licenses, the 237 decomposition,
  scenarios, dataset/prompt provenance, and data hosting.

## Codebase internals

- [`codebase/overview.md`](codebase/overview.md) — architecture portal.
