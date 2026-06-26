# FLiPS reproduction Makefile — Mode A (no-GPU) wiring across three honest tiers.
#
#   out-of-box  : zero user upload, zero download, CPU — runs against the committed
#                 in-repo light subset (Productions/FLiPS_ICML_light_subset/) and the
#                 shipped LLMmap cache.
#   data-gated  : needs the user-uploaded headline run Productions/$(RUN_NAME)/ (the
#                 off-disk FLiPS_ICML_run) — these targets SKIP LOUDLY (exit 0) when it
#                 is absent, they never crash with FileNotFoundError.
#   Mode-B      : GPU from-scratch path (see docs/reproduction/mode-b.md).
#
# All paths are relative to the repo root; no machine-specific absolute paths.
# Override on the command line, e.g.  make repro-closedset RUN_NAME=my_run PYTHON=python3
#
# ENV: targets that classify need `audit_llm` importable — run inside the project's
# poetry env (`.venv`) or pass an interpreter that has it, e.g. PYTHON=.venv/bin/python.
# A bare system `python` makes the XP child process die with `ModuleNotFoundError:
# No module named 'audit_llm'` (run_experiments adds src/ to the dispatcher path only).
#
# `make` with no target prints `help`.

PYTHON   ?= python
RUN_NAME ?= FLiPS_ICML_run
SCENARIO ?= main

# Mode-B smoke: one small model, end-to-end (guarded — SKIPs without GPU + vLLM).
MODEB_SMOKE_MODEL   ?= Qwen/Qwen2-1.5B-Instruct
MODEB_SMOKE_DATASET ?= FLiPS_ICML

# The committed, zero-download CPU fixture (never overridden — out-of-box targets bind to it).
LIGHT_RUN := FLiPS_ICML_light_subset

# Reproduction config dirs under XP_configs/ (reconciled against the post-03 repo).
LLMMAP_DIR     := e3_llmmap_baseline
COMPARISON_DIR := e3_flips_vs_llmmap
CLOSEDSET_DIR  := e1_closedset_headline
OPENSET_DIR    := e2_openset_headline
ABLATIONS_DIR  ?= Ablations
SMOKE_DIR      := Smoke_light

RUN_XP := $(PYTHON) XP_configs/run_experiments.py

# --------------------------------------------------------------------------- #
# Optional detached / deferred execution (long data-gated targets)             #
#   make repro-closedset SCREEN=1           # detached GNU screen, session = target name
#   make repro-closedset SCREEN=myrun       # detached, custom session name
#   make repro-headline  SCREEN=1 DELAY=2   # wait 2h (inside the screen) then run the chain
# DELAY (hours, decimals ok) implies SCREEN. Survives an SSH/Jupyter disconnect; the sleep
# runs inside the screen, not your terminal. Logs to Productions/_screenlogs/<session>.log.
# The re-entered child carries FLIPS_IN_SCREEN=1 (recursion guard) so it does the real work.
# --------------------------------------------------------------------------- #
SCREEN ?=
DELAY  ?= 0
# Recipe-`if` condition: a detached run is requested AND we are not already the in-screen child.
screen_wanted = { [ -n "$(SCREEN)" ] || [ "$(DELAY)" != "0" ]; } && [ -z "$$FLIPS_IN_SCREEN" ]
# Re-launch THIS goal ($@) detached; clear SCREEN/DELAY for the child (FLIPS_IN_SCREEN guards too).
screen_launch = bash scripts/screen_run.sh "$@" "$(SCREEN)" "$(DELAY)" $(MAKE) $@ SCREEN= DELAY=0

.DEFAULT_GOAL := help

.PHONY: help fetch fetch-check fetch-headline gate-a gate-golden gate-b gate-b-parity \
        repro-llmmap-e3 smoke example-scenario repro-closedset repro-openset repro-comparison \
        repro-ablations repro-headline figures repro-all clean repro-modeb-smoke infer-smoke

# --------------------------------------------------------------------------- #
# Data-gated guard: run only if the headline run_config is present, else SKIP   #
# loudly (exit 0). Guard + run live in ONE shell line so the SKIP short-circuits#
# the run command (each Make recipe line is otherwise its own shell).          #
#   $(call repro_gated,<config-dir>,<xp-suffix>)                               #
# --------------------------------------------------------------------------- #
define repro_gated
@if $(screen_wanted); then \
		$(screen_launch); \
	elif [ -f "Productions/$(RUN_NAME)/run_config.json" ] || [ -f "Productions/$(RUN_NAME)/run_config.pickle" ]; then \
		echo ">>> [data-gated] $(1): headline run Productions/$(RUN_NAME)/ present — running."; \
		$(RUN_XP) --config-dir $(1) --run-name $(RUN_NAME) --xp-suffix $(2) --mode inline; \
	else \
		echo ">>> [data-gated] $(1): needs the user-uploaded FLiPS_ICML_run headline run."; \
		echo ">>>   Productions/$(RUN_NAME)/run_config.{json,pickle} not found — run \`make fetch-headline\` first; SKIPPING."; \
	fi
endef

help:
	@echo "FLiPS reproduction targets (RUN_NAME=$(RUN_NAME), SCENARIO=$(SCENARIO), PYTHON=$(PYTHON))"
	@echo ""
	@echo "  Data fetch:"
	@echo "    fetch            download the released archive(s) from Zenodo            [out-of-box wiring]"
	@echo "    fetch-check      verify present files against data/manifest.sha256        [out-of-box]"
	@echo "    fetch-headline   download the off-disk FLiPS_ICML_run headline run        [data-gated]"
	@echo ""
	@echo "  Out-of-box (zero download, CPU, runs on the committed light subset):"
	@echo "    gate-a           always-runs gate: LLMmap E3 replot + light-subset smoke  [out-of-box]"
	@echo "    gate-golden      golden regression: re-run smoke, assert vs golden_refs   [out-of-box]"
	@echo "    gate-b-parity    data-free class-parity firewall (enumerator vs legacy)   [out-of-box]"
	@echo "    gate-b           parity firewall + accuracy sanity (headline part SKIPs)  [out-of-box+data-gated]"
	@echo "    repro-llmmap-e3  the LLMmap-only F01 E3 curve (no upload needed)          [out-of-box]"
	@echo "    smoke            zero-download CPU smoke classification, emits >=1 PDF     [out-of-box]"
	@echo "    example-scenario run the shipped 'run your own scenario' template          [out-of-box]"
	@echo ""
	@echo "  Data-gated (need the user-uploaded headline run; SKIP loudly if absent):"
	@echo "    repro-closedset  closed-set E1 headline (96%)                             [data-gated]"
	@echo "    repro-openset    open-set E2 headline (90%)                               [data-gated]"
	@echo "    repro-comparison 3-curve E3 (FLiPS closed/open vs LLMmap)                 [data-gated]"
	@echo "    repro-ablations  ablation sweep A1-A9 (configs ship with the headline)    [data-gated]"
	@echo "    repro-headline   closedset -> openset -> comparison -> gate-b, in order    [data-gated]"
	@echo ""
	@echo "  Detached / deferred (long data-gated targets; survives SSH/Jupyter disconnect):"
	@echo "    SCREEN=1         run the target in a detached GNU screen (session = target name)"
	@echo "    SCREEN=<name>    ... with a custom session name"
	@echo "    DELAY=<hours>    wait N hours INSIDE the screen before starting (implies SCREEN)"
	@echo "      e.g.  make repro-headline SCREEN=1 DELAY=2   (logs to Productions/_screenlogs/)"
	@echo ""
	@echo "  Figures / orchestration / cleanup:"
	@echo "    figures          re-plot micro-PR + open-set ROC from cached pickles      [out-of-box/data-gated]"
	@echo "    repro-all        fetch -> gate-a -> repro-llmmap-e3 -> data-gated -> figures"
	@echo "    clean            remove generated outputs under Productions/\$$(RUN_NAME)   (NOT the light subset)"
	@echo ""
	@echo "  Mode B (GPU, from scratch) — full path in docs/reproduction/mode-b.md:"
	@echo "    repro-modeb-smoke  one small model end-to-end (SKIPs without GPU + vLLM)   [Mode-B]"

# --------------------------------------------------------------------------- #
# Data fetch                                                                   #
# --------------------------------------------------------------------------- #
fetch:
	$(PYTHON) scripts/fetch_data.py

fetch-check:
	$(PYTHON) scripts/fetch_data.py --check-only

fetch-headline:
	$(PYTHON) scripts/fetch_data.py --with-headline

# --------------------------------------------------------------------------- #
# Out-of-box tier                                                             #
# --------------------------------------------------------------------------- #
gate-a:
	$(PYTHON) scripts/gates/gate_a.py

gate-golden:
	$(PYTHON) scripts/gates/gate_golden.py

# Data-free class-parity firewall (always runs). part (a) of gate-B.
gate-b-parity:
	$(PYTHON) scripts/gates/gate_b_parity.py

# Full gate-B: parity firewall (a, always) + coarse accuracy sanity (b). The headline
# accuracy check SKIPs loudly (never falsely passes) when Productions/$(RUN_NAME)/ is absent.
gate-b:
	$(PYTHON) scripts/gates/gate_b.py --run-name $(RUN_NAME)

repro-llmmap-e3:
	$(PYTHON) scripts/gates/gate_a.py --llmmap-only

smoke:
	$(RUN_XP) --config-dir $(SMOKE_DIR) --run-name $(LIGHT_RUN) --xp-suffix smoke --mode inline

# "Run your own scenario" template: a declarative scenario (config/scenarios/example_custom.yaml)
# selecting a 4-class subset of the committed light subset. Zero download, CPU. Copy the two
# example files and edit them to run your own. Full guide: docs/reproduction/custom-scenario.md.
example-scenario:
	$(RUN_XP) --config-dir example_custom --run-name $(LIGHT_RUN) --xp-suffix example --mode inline

# --------------------------------------------------------------------------- #
# Data-gated tier (SKIP loudly when the headline run is absent)               #
# --------------------------------------------------------------------------- #
repro-closedset:
	$(call repro_gated,$(CLOSEDSET_DIR),closedset)

repro-openset:
	$(call repro_gated,$(OPENSET_DIR),openset)

repro-comparison:
	$(call repro_gated,$(COMPARISON_DIR),comparison)

repro-ablations:
	@if $(screen_wanted); then \
		$(screen_launch); \
	elif [ ! -d "XP_configs/$(ABLATIONS_DIR)" ]; then \
		echo ">>> [data-gated] repro-ablations: no XP_configs/$(ABLATIONS_DIR)/ in this repo."; \
		echo ">>>   ablation configs (A1-A9) ship with the headline release; SKIPPING."; \
	elif [ -f "Productions/$(RUN_NAME)/run_config.json" ] || [ -f "Productions/$(RUN_NAME)/run_config.pickle" ]; then \
		$(RUN_XP) --config-dir $(ABLATIONS_DIR) --run-name $(RUN_NAME) --xp-suffix ablations --mode inline; \
	else \
		echo ">>> [data-gated] repro-ablations: needs the user-uploaded FLiPS_ICML_run headline run."; \
		echo ">>>   Productions/$(RUN_NAME)/run_config.{json,pickle} not found — run \`make fetch-headline\` first; SKIPPING."; \
	fi

# Full headline reproduction chain, sequential (one at a time — each saturates CPU/RAM).
# Screen-/delay-aware: `make repro-headline SCREEN=1 DELAY=2` queues the whole chain to
# start in 2h, detached. gate-b runs LAST (it reads the results.json the repros emit).
# `set -e` so a real failure stops the chain; a data-absent SKIP is exit 0 and continues.
repro-headline:
	@if $(screen_wanted); then \
		$(screen_launch); \
	else \
		set -e; \
		$(MAKE) repro-closedset; \
		$(MAKE) repro-openset; \
		$(MAKE) repro-comparison; \
		$(MAKE) gate-b; \
	fi

# --------------------------------------------------------------------------- #
# Figures — re-plot from caches produced by a run (smoke/repro) or fetched     #
# --------------------------------------------------------------------------- #
figures:
	@found=0; \
	for c in $$(find Productions -name micro_pr_curve_cache.pkl 2>/dev/null); do \
		echo ">>> micro-PR curve from $$c"; \
		$(PYTHON) scripts/fig_scripts/preview_micro_pr_curve_cached.py --cache "$$c" && found=1; \
	done; \
	for c in $$(find Productions -name roc_figscore_cache.pkl 2>/dev/null); do \
		echo ">>> open-set ROC from $$c"; \
		$(PYTHON) scripts/fig_scripts/preview_openset_roc.py --cache "$$c" && found=1; \
	done; \
	if [ "$$found" = "0" ]; then \
		echo ">>> no *_cache.pkl found under Productions/ — run \`make smoke\` or a \`make repro-*\`"; \
		echo ">>>   (or fetch the archive) first; nothing to plot."; \
	fi

# --------------------------------------------------------------------------- #
# Mode B (GPU, from scratch) — guarded one-model inference smoke.              #
# Prints "needs GPU + vLLM" and exits 0 when torch/vllm aren't importable, so  #
# it is safe to run on a CPU-only build/CI box. Full path: mode-b.md.          #
# --------------------------------------------------------------------------- #
repro-modeb-smoke infer-smoke:
	@if $(PYTHON) -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('torch') and importlib.util.find_spec('vllm') else 1)" 2>/dev/null; then \
		echo ">>> [Mode-B] torch + vLLM present — running one-model inference smoke."; \
		echo ">>>   model=$(MODEB_SMOKE_MODEL) dataset=$(MODEB_SMOKE_DATASET)"; \
		$(PYTHON) scripts/Run_Inferences.py --dataset $(MODEB_SMOKE_DATASET) --model $(MODEB_SMOKE_MODEL) --sub_run modeb_smoke --gpu 1 --parse_gen --seed 42; \
	else \
		echo ">>> [Mode-B] repro-modeb-smoke needs GPU + vLLM, which are not importable here."; \
		echo ">>>   Install: poetry install --extras generation, then a matching vLLM build (see docs/reproduction/mode-b.md)."; \
		echo ">>>   SKIPPING (this is expected on a CPU-only box)."; \
	fi

# --------------------------------------------------------------------------- #
# Orchestration + cleanup                                                      #
# --------------------------------------------------------------------------- #
repro-all:
	@$(MAKE) fetch || echo ">>> fetch skipped (no real DOI wired / offline) — continuing with in-repo data."
	@$(MAKE) gate-a
	@$(MAKE) repro-llmmap-e3
	@$(MAKE) repro-closedset
	@$(MAKE) repro-openset
	@$(MAKE) repro-comparison
	@$(MAKE) figures

clean:
	@if [ "$(RUN_NAME)" = "$(LIGHT_RUN)" ]; then \
		echo ">>> refusing to clean the committed light subset ($(LIGHT_RUN)); set RUN_NAME=<fetched run>."; \
	else \
		echo ">>> cleaning generated outputs under Productions/$(RUN_NAME) (NOT the committed light subset)"; \
		rm -rf "Productions/$(RUN_NAME)/Experiments/Batch_Classification_across_token_pairs"; \
		rm -rf "Productions/$(RUN_NAME)/Analysis/Graph_Numeric/Figures"; \
		find "Productions/$(RUN_NAME)" -name '*_cache.pkl' -delete 2>/dev/null || true; \
	fi
	@rm -rf tmp/micro_pr_curve_preview tmp/openset_roc_preview
	@rm -rf xp_logs
