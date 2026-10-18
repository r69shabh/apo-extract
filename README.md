# apo-extract

Automated prompt optimization for structured extraction from PDFs, evaluated on the
as-published [ExtractBench](data/extract-bench) schemas (hiring/resume,
finance/credit_agreement, academic/research, sport/swimming).

Pipeline: deterministic seed split → baseline extraction with a Codified-Context
prompt artifact → failure diagnosis (subtree × error-type clusters) → SR-PBO loop
(candidate proposals, staged filtering, Pareto archive with discretized primary
metric, tabu + delta-dedup, subtree rollback) → held-out test evaluation and a
generated report.

## Install

```bash
uv sync          # Python 3.12 pinned in .python-version; extract-bench vendored in data/
```

## Offline checks (no API key)

```bash
uv run pytest -q                                  # full suite, fully mocked
uv run apo-extract dry-run -c configs/hiring_resume.yaml   # gold-as-prediction sanity: test F1 must be 1.0
```

## Live run (Gemini)

```bash
export GEMINI_API_KEY=...
uv run apo-extract run     -c configs/hiring_resume.yaml
uv run apo-extract resume  -c configs/hiring_resume.yaml   # after Ctrl-C/reboot; cached work is reused
uv run apo-extract inspect --run-id resume-srpbo-v1 --print-prompt
uv run apo-extract evaluate -c configs/hiring_resume.yaml --run-id resume-srpbo-v1
uv run apo-extract diff     -c configs/hiring_resume.yaml --run-id resume-srpbo-v1
uv run apo-extract report   --run-id resume-srpbo-v1
```

Everything a run produces lives under `runs/<run_id>/`:

```
state.db            SQLite (WAL): runs, llm_calls, predictions_cache, candidates,
                    iterations, archive, tabu
llm_calls.jsonl     append-only call log (tokens, cost, latency, raw paths)
raw/                raw request/response JSON per LLM call
iterations/iter_N/  per-candidate train/val evaluation dumps
final/              best_artifact.json, summary.json, test_per_document.json
report.md           generated run report (iterations, hypotheses, subtree deltas)
cache/              PDF sha256 manifest + Gemini File API upload cache
```

## Frontend (run workbench)

`frontend/` is a Vite + React + shadcn/ui workbench that renders an exported
run snapshot: seed vs. optimized prompt, KPI strip, iteration ledger, score /
Pareto charts, per-subtree scoreboard, plus a "how it works" section with SVG
schematics of the loop, scoring attribution, gates, dedup and crash-safety.

```bash
uv run apo-extract export-web --run-id <run-id>   # writes frontend/public/data/*.json
cd frontend && bun install && bun run dev          # http://localhost:5173
```

Re-run `export-web` any time (even while the run is live) to refresh the page.

## Config

All knobs live in `configs/*.yaml` (validated by `src/apo_extract/config.py`):
dataset paths + split policy, scoring engine + array policy + judge model, LLM
roles (extractor / mutator / judge), budgets (iterations, USD, tokens, wall clock)
and optimizer safeguards (beam, stage-1 gate, Pareto tolerance, stall patience,
duplicate similarity, rollback). Retargeting a schema or swapping models requires
no code changes; `scoring.engine` also accepts `your.module:factory` to replace
the scorer entirely.

## Layout

```
src/apo_extract/
  config.py            pydantic config schema
  dataset/             loader (schema + pdf+gold triplets), deterministic splitter
  llm/                 litellm client (+Gemini File API path), PDF handler, telemetry
  predictor/           PromptArtifact, Extractor (single-call / subtree-chunked)
  scoring/             extract-bench-compliant engine, metric registry, alignment, cache
  optimizer/           diagnostics, proposer (+crossover), safeguards, SR-PBO loop
  observability/       SQLite store, diff tool, report generator
  cli.py               run / resume / dry-run / evaluate / diff / inspect / report / export-web
```

## Compliance notes (differences from the blueprint, and why)

- The vendored extract-bench repo is the ground truth for metric semantics; the
  blueprint's simplified formulas were adapted where they disagreed:
  `number_tolerance` is extract-bench's absolute (effectively binary) tolerance;
  the array metric reports soft-F1 over Hungarian-matched items by default, with
  `array_alignment_policy: extract_bench_default` restoring the upstream LLM-judge
  `matched/(matched+missed)` behavior; per-node P/R/F1 uses the attribution scheme
  in `scoring/engine.py` (absent-gold/absent-pred = TN, excluded).
- `string_semantic` / `string_llm` / `array_llm` are LLM-judge metrics: they need
  a working key and use `scoring.judge_model` (litellm id) via metric params.
  Offline test runs and `dry-run` override them with exact matching.
- Split allocation: `n_test = max(1, floor(test_ratio*N))`,
  `n_val = max(2, floor(val_ratio*N))`, rest to train, after a seeded shuffle
  (`dataset.split_policy.seed`), so runs are reproducible.
- Deduplication compares each mutation's *delta text per target* — whole-prompt
  similarity would reject every incremental edit (siblings share ~95% of text).
