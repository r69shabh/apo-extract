# REPORT — Automated Prompt Optimization for Structured Extraction (ExtractBench)

Status: implementation complete and fully validated offline. **Live Gemini
evaluation numbers are pending a run with `GEMINI_API_KEY`** (see Run commands);
all reported-in-code claims below are verified by the offline test suite.

## 1. Methodology

- **Prompt representation**: `PromptArtifact` = global preamble, formatting/null
  policy, per-top-level-property `subtree_directives` (`$.field`), verification
  checklist. Mutations are additive, surgical, and versioned (`artifact_id`,
  `parent_ids`, `generation`); `hash()` identifies prompt versions across caches.
- **Scoring**: `ExtractBenchCompliantScorer` uses the vendored extract-bench
  metric registry with published evaluation configs/presets. Per-node P/R/F1
  attribution: absent-gold/absent-pred = TN (excluded); gold absent, pred present
  = FP; pred absent (gold present) = FN; both present = TP = score, FN += 1 − score.
  Arrays use deterministic Hungarian alignment (soft-F1 by default; the upstream
  LLM-judge policy is selectable). LLM-judge metrics are disk-cached.
- **Diagnosis**: failures clustered by (top-level subtree × error type); cluster
  weight = count × (1 + (1 − subtree F1)); highest-weight cluster is targeted.
- **Optimization (SR-PBO)**: each iteration proposes k candidate mutations against
  the incumbent (LLM mutator when configured; deterministic templates otherwise),
  applies stage-1 filtering (train regression cap 0.02), evaluates survivors on
  validation, and maintains a Pareto archive over (val micro-F1, macro-subtree F1)
  with discretized tie bands (0.03). Tabu list + per-target delta dedup prevent
  cycling; crossover merges complementary front directives; a rollback pass reverts
  subtrees that regressed > 0.05 while global score improved. Stops on stall
  patience or budget (iterations/tokens/USD/wall-clock). Test split is scored once,
  at the end, for the best artifact only.
- **Reproducibility**: seeded shuffle split; all prompts, predictions, scores,
  hypotheses and LLM payloads persisted under `runs/<id>/`; resume reuses cached
  (prompt, PDF, model) predictions and completed iterations.

## 2. Experimental setup

- Datasets (as published, unmodified): hiring/resume (7 docs), finance/
  credit_agreement (10), academic/research (9), sport/swimming (N as shipped) —
  schemas and `pdf+gold` triplets under `data/extract-bench/dataset/`.
- Splits: seeded 30/40/30 with `max()` floors (e.g. resume: train 3 / val 2 / test 2).
- Models: extractor `gemini-3.8-flash` (temperature 0); judge metrics
  `gemini/gemini-3.8-flash` via `scoring.judge_model`; mutator LLM optional.
- Budgets per `configs/*.yaml`: 12 iterations, caps in tokens/USD/time.

## 3. Offline verification (done)

| Check | Result |
|---|---|
| Gold-vs-gold scoring, all 5 shipped schemas, real evaluation configs | F1 = 1.000 (with deterministic leaf-metric overrides) |
| JSON-unparseable prediction | recall 0, `json_valid=false`, whole-doc omission cluster |
| Array item dropped / invented | FN = 1 / FP = 1 exactly; item-level P/R |
| Ambiguous array alignment | Tier-2 judge refinement only in 0.30 < s < 0.85 band |
| Repeat scoring determinism | identical results across runs (incl. cached judges) |
| Splitter | seeded, reproducible, disjoint, blueprint N=5/6/7/10 allocation |
| Optimizer dry run (directive-dependent mock, real resume dataset) | learns missing directive; val F1 1.0; crossover accepted; hypotheses logged |
| Crash + resume (KeyboardInterrupt mid-iteration 2, then resume) | zero re-extractions of cached (prompt, PDF) pairs; run completes |
| Subtree rollback | regressed subtrees reverted to parent directive; skips below threshold |
| `dry-run` CLI (gold-as-prediction) with production scorer wiring | test F1 = 1.0000 |
| Full test suite | 68 tests, all green |
| Retry hardening (429 log, bounded attempts, fail-fast on daily-quota delays) | covered by `tests/test_cli_retry.py` |
| Unknown error-kind template fallback | covered by `tests/test_proposer_templates.py` |
| Text-mode PDF ingestion for non-vision providers (`pdf_mode: text`) | covered by `tests/test_client_attachments.py` |

## 4. Live results

From `runs/<id>/final/summary.json` + `report.md`. Run with e.g.:

```bash
source ~/.zshrc  # GEMINI_API_KEY / OPENROUTER_API_KEY
uv run apo-extract run -c configs/hiring_resume.yaml
# repeat for configs/finance_credit_agreement.yaml, configs/academic_research.yaml, configs/swimming.yaml
```

| Dataset | Seed artifact (test F1) | Optimized (test F1) | Δ | Iterations accepted | Cost (USD) |
|---|---|---|---|---|---|
| hiring/resume | 0.7157 | 0.7221 | +0.0064 | 3 | 0.44 (est. tokens; ≈$0.03 actual, OpenRouter) |
| finance/credit_agreement | pending | pending | | | |
| academic/research | pending | pending | | | |
| sport/swimming | 1.0000 | 1.0000 | +0.0000 | 5 | 0.00 (Groq free tier; 283k tokens) |

Runs 3–4 (`finance/credit_agreement`, `academic/research`) are fully staged
(`configs/*.yaml` wired to `openrouter/google/gemini-2.5-flash-lite`, JPEG
page-image ingestion for the 97–218-page PDFs, sequential driver at
`runs/autopilot.sh`) but remain blocked on account credits: the OpenRouter
account is free-tier with its trial allowance exhausted, and Gemini's direct
free-tier daily caps (~20 requests/day) cannot cover a full run. Estimated cost
to unblock both: ~$0.8–1.2. Run 2 (sport/swimming) fit inside Groq's free tier
and completed.

Per-run analysis (subtree deltas, accepted hypotheses, rollback events) is
auto-generated in `runs/<id>/report.md`.

Run-1 (hiring/resume) methodology notes — deviations forced by free-tier caps:

- Iterations 1–5 ran on `gemini-3.5-flash` → `gemini-3.5-flash-lite` (paid-tier
  daily quotas exhausted mid-run; exact model per iteration is in
  `runs/resume-srpbo-v1/llm_calls.jsonl`).
- The held-out finalize (val re-eval + test) ran on
  `openrouter/google/gemini-2.5-flash-lite` (same family) with page-image PDF
  ingestion (`pdf_mode: images`) and judge `openrouter/google/gemini-2.5-flash-lite`;
  the seed's test F1 above was measured under identical conditions for a fair Δ.
- NVIDIA NIM (`z-ai/glm-5.3-flash`) was tried as a no-quota fallback and rejected:
  serial extraction-size call took 568 s and returned empty content.
- Budget: `max_iterations` 5 (shipped default 12); subtree rollback disabled for the
  resume run (resume-time staging to skip a doomed iteration; see config comments).
- Test macro subtree F1 moved 0.7138 → 0.5340: the accepted `$.workExperience`
  directive fixed `$.skills` (0 → 0.71) but test-split `$.other` collapsed on 2 docs
  (val `$.other` did too: 0.62 → 0.0) — tiny test set, treat macro deltas as noisy.

Run-2 (sport/swimming) methodology notes — constraints from Groq's free tier:

- Groq's per-model daily cap is 200k tokens: `gpt-oss-120b` ran iterations 1–4
  then hit TPD (199,196/200,000 — "try again in ~10m" rolling recovery). All
  three roles were switched to `qwen/qwen3.8-27b` (fresh daily bucket) at
  iteration 4; the archive was re-scored because prediction/judge caches are
  model-keyed. Val F1 trajectory mixes both models (0.88 → 0.94 → 0.97 → 0.99);
  the finalize passes (val re-eval + test for seed and best) ran on qwen only,
  and the seed's test F1 above was measured under identical conditions.
- Groq caps input + `max_tokens` at 8k tokens (TPM) per request;
  `max_output_tokens` are set to 3000 (extractor) / 2000 (mutator) / 1000
  (judge). A transiently lower 1200-token extractor cap (mid-run rate-limit
  tuning) truncated one test document during the finalize — see the ceiling
  bullet below. One empty generation (Groq `json_validate_failed`) killed the
  run once before this was treated as transient by the bounded retry wrapper
  (`src/apo_extract/cli.py`).
- The run converged early: stall patience 2 stopped it at iteration 5 of 12
  (5 candidates accepted; best c5_0, val F1 0.9928, mutated directive 65 → 391
  bytes).
- Ceiling effect: the test split is a single document and the seed already
  extracts it perfectly — both arms score test F1 1.0000 when re-measured with
  a full 3000-token output cap. (The run's own finalize, executed under the
  free-tier 1200-token cap set mid-run, scored both arms 0.9636: the misses
  were output truncation, not extraction errors. `summary.json` keeps that
  finalize pass as `finalize_test_f1` and the identically-measured pair as
  `test_f1`/`seed_test_f1`; the table above uses the pair.) Accepted
  directives moved val (0.88 → 0.99), not test.
- Cost: $0.00 (Groq free tier), 283k tokens spent.
- Resume replay artifact: mid-run rate-limit kills left iteration 5 half-recorded;
  a resumed attempt re-rolled its slots and overwrote the `c5_0` ledger row with
  a rejected re-proposal (train F1 0.9675 < gate). The accepted original stayed
  in the archive and is the `best_candidate_id` in `summary.json` (val 0.9928),
  so the iteration ledger understates `c5_0` — read its row as the re-proposal.
  Fixed forward in `_run_iteration` (replay now fills only unrecorded slots and
  skips fully-recorded iterations; `tests/test_resume_replay.py`), so the
  remaining runs won't hit this.

## 5. Limitations & future work

- LLM-judge leaf metrics make scoring cost scale with string-field count; the disk
  cache mitigates repeats across candidates but not the first full pass.
- Validation sets are tiny (2 docs on resume) — variance is real; the archive's
  aggregate-F1 dimension and train gate reduce overfitting to val noise.
- Rollback operates on top-level subtrees only (matches directive granularity).
- The mutator LLM path is implemented but exercised only against mocked responses;
  live-run hypothesis quality is part of the pending evaluation.
