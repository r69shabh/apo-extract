# REPORT — Automated Prompt Optimization for Structured Extraction (ExtractBench)

`apo-extract` optimizes a single extraction prompt (`PromptArtifact`: global
preamble, formatting/null policy, per-subtree directives, verification checklist)
against the vendored ExtractBench benchmark. Loop: subtree×error-type diagnosis →
LLM mutation candidates → stage-1 train gate → validation → Pareto archive over
(val micro-F1, macro-subtree F1, tie band 0.03) → rollback; test split scored once,
at the end, for the best artifact only. Two live runs completed; two remain staged
but blocked on credits (see Limitations). All artifacts (`summary.json`,
`report.md`, ledger DB, best artifact, per-document test evidence) are under
`runs/<id>/final/`.

**Run conditions** (the numbers below are only meaningful under these):

- **hiring/resume** (7 docs; train 3 / val 2 / test 2): iterations on
  `gemini-3.5-flash`→`gemini-3.5-flash-lite` (free tiers), finalize + seed
  re-measure on `openrouter/google/gemini-2.5-flash-lite`, `pdf_mode: images`;
  judge = same model; 5 of 12 iterations (stall); rollback off.
- **sport/swimming** (train 2 / val 2 / test 1): iterations on `gpt-oss-120b` →
  `qwen/qwen3.8-27b` (Groq 200k TPD exhausted mid-run; caches are model-keyed, so
  the archive was re-scored after the switch); finalize + seed re-measure on
  `qwen/qwen3.8-27b`, `pdf_mode: text`; judge = same model; 5 of 12 iterations
  (stall patience 2); rollback on.

## 1. Test-set scores (seed vs final)

| Run | Test docs | Seed test F1 | Final test F1 | Δ (micro) | Seed→final macro-F1 |
|---|---|---|---|---|---|
| hiring/resume | 2 | 0.7157 | **0.7221** | +0.0064 | 0.7138 → 0.5340 |
| sport/swimming | 1 | 1.0000 | **1.0000** | +0.0000 | 1.0000 → 1.0000 |

Per-subtree test F1 (seed → final), top-level subtrees from the merged
per-document evidence:

**hiring/resume**

| Subtree | Seed | Final | Δ |
|---|---|---|---|
| $.certificationsAndAwards | 0.537 | 0.824 | **+0.287** |
| $.socialLinks | 1.000 | 1.000 | +0.000 |
| $.skills | 0.000 | 0.000 | +0.000 |
| $.personalInfo | 0.956 | 0.898 | −0.058 |
| $.workExperience | 0.780 | 0.698 | −0.081 |
| $.education | 0.933 | 0.636 | −0.296 |
| $.other | 0.390 | 0.000 | −0.390 |
| $.publications | 0.619 | 0.000 | −0.619 |
| $.media | 0.909 | 0.000 | −0.909 |
| $.languages | 1.000 | 0.000 | −1.000 |

**sport/swimming**

| Subtree | Seed | Final | Δ |
|---|---|---|---|
| $.age_groups | 1.000 | 1.000 | +0.000 |
| $.championship | 1.000 | 1.000 | +0.000 |
| $.event_details | 1.000 | 1.000 | +0.000 |

Read the resume table with care: the micro gain is real but concentrated
(`$.certificationsAndAwards`); the collapses are mostly subtrees with 0–2 test
instances (e.g. `$.languages` is one sparse field), and the val split shows the
intended repair the test split does not (`$.skills` 0 → 0.714 on val). Defensible
claim: the run improved one field family and did not degrade the aggregate micro
metric; per-subtree test deltas on 2 documents are noise-dominated.

Swimming note: both arms are at the ceiling on the single test document
(re-measured with the final 3000-token output cap). The run's own finalize,
executed mid-run under a transient 1200-token cap, scored both arms 0.9636 —
those misses were output truncation, not extraction errors. `summary.json` keeps
the finalize pass as `finalize_test_f1` and the re-measured pair as
`test_f1`/`seed_test_f1`; this table uses the pair.

## 2. Prompt diff (seed → final)

Both winners differ from the seed **only in one subtree directive**; preamble,
formatting policy, checklist and all other directives are byte-identical to seed.

**hiring/resume** — final = seed + this `$.workExperience` directive
(5 accepted mutations total; the directive accumulated through the run):

> For the `category` field, strictly output null unless the document explicitly
> contains a designated category tag associated with the work entry; do not use
> section headers or headings as category values.
> For array fields, extract one entry per real item and copy each entry's values
> exactly as written. Do not merge two items into one, split one item into
> several, reorder, or drop entries that do not match an example.

Trigger: diagnosis clustered 6 array_item_mismatch failures on
Resume-Academic02 / Resume-Legal (val F1 0.7264 → 0.7316, aggregate 0.7777 → 0.7843).

**sport/swimming** — final = seed + this `$.age_groups` directive (65 → 391 bytes):

> CRITICAL: For the 'results' array, ensure strict 1:1 mapping between each row
> in the document table and an object in the array. Do not skip rows where
> specific attributes (like 'records') are missing in the source; instead,
> represent missing data as null (e.g., "records": null) to maintain array
> length and index alignment.

Note: several accepted swimming candidates edited the preamble/checklist
(iterations 1–4); the Pareto winner is the **directive-only** variant (val 0.9928
vs 0.9722 for the best checklist variant), so those edits are not in the final
prompt. "Accepted" means "admitted to the front", not "kept".

## 3. Notable accepted mutations

**hiring/resume** (val F1 = 0.7264 seed baseline):

| It. | Candidate | Mutation | Target | Val F1 | Hypothesis (mutator) / outcome |
|---|---|---|---|---|---|
| 1 | c1_0 | directive_array_item_mismatch | $.workExperience | 0.7511 | Add array-item guidance to fix 6 observed failures (Resume-Academic02, Resume-Legal); accepted (+0.025) |
| 2 | c2_0 | directive_array_item_mismatch | $.workExperience | 0.7511 | Refinement of the same rule; accepted (tie) |
| 3 | c3_0 | directive_array_item_mismatch | $.workExperience | 0.7316 | Third refinement; accepted as **final best** (tie band + aggregate 0.7843) |
| 4 | c4_0 | directive_array_item_mismatch | $.certificationsAndAwards | — | Add guidance for 4 observed failures; rejected stage-1 (train regression) |
| 5 | c5_0 | directive_array_item_mismatch | $.certificationsAndAwards | — | Same target; rejected stage-1 |

**sport/swimming** (val F1 = 0.9865 seed baseline, model qwen):

| It. | Candidate | Mutation | Target | Val F1 | Hypothesis (mutator) / outcome |
|---|---|---|---|---|---|
| 1 | c1_0 | llm_preamble_edit | global_preamble | 0.8835 | Replace placeholder rank values ("NA") with null, eliminating rank mismatches; accepted to front |
| 1 | c1_1 | llm_checklist_edit | verification_checklist | 0.9393 | Explicit checklist reminder reinforces placeholder→null rule; accepted |
| 3 | c3_1 | llm_checklist_edit | verification_checklist | 0.9722 | Checklist reminder, second refinement; accepted |
| 4 | c4_0 | llm_directive_edit | $.age_groups | 0.9865 | 1:1 row↔object mapping to stop dropped/extra rows in results tables; accepted |
| 4 | c4_1 | llm_checklist_edit | verification_checklist | 0.9865 | Array-length consistency check; accepted |
| 5 | c5_0 | llm_directive_edit | $.age_groups | **0.9928** | Strict 1:1 mapping with nulls for missing attributes; accepted as **final best** |

(Swimming iteration-2/3 proposals were rejected as dominated; the run stopped at
iteration 5 by stall patience 2. The public ledger for `c5_0` shows a rejected
re-proposal row — a replay artifact of mid-run rate-limit kills, fixed forward;
the accepted original is in the archive and is `best_candidate_id`.)

## 4. Limitations (as implemented)

- **Held-out splits are tiny** (test: 1–2 docs). Per-subtree and macro test
  deltas are variance-dominated; the val split is the more informative signal,
  and even it is 2 docs. Do not read resume's macro collapse (or swimming's ±0)
  as a property of the directives.
- **Free-tier constraints shaped both runs**: mid-run model switches (both runs),
  8k-TPM request sizing (`max_output_tokens` 3000/2000/1000), one truncation
  incident (swimming finalize), and model-keyed caches meaning a switch re-scores
  the archive — val values from different iterations are approximately, not
  exactly, comparable.
- **Two of four dataset runs are missing**: finance/credit_agreement (10 docs,
  42–130k tokens/doc) and academic/research (9 docs, 4–218k tokens/doc) exceed
  Groq free-tier context and need ~$0.8–1.2 of OpenRouter credits (configs and
  driver staged: `configs/*.yaml`, `runs/autopilot.sh`). Large-document schemas
  are the untested regime.
- **"Accepted" ≠ "in final prompt"**: the Pareto winner retains a subset of
  accepted edits (see §2 swimming note). Cost accounting is token-estimated for
  the resume run (≈$0.03 actual on OpenRouter light tier, 326k tokens).
- **Scoring noise**: semantic/array fields go through an LLM judge (disk-cached,
  model-keyed); judge model changes re-judge. Rollback operates at top-level
  subtree granularity only.
- **Operational**: resume has no config-hash validation (documented staging
  hazard); the ledger replay artifact above (fixed in `_run_iteration`, tests in
  `tests/test_resume_replay.py`); the mutator LLM path has only these two live
  runs of track record.

## Reproduce

```bash
source ~/.zshrc  # GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY
uv run apo-extract run -c configs/swimming.yaml       # free tier, ~5 iterations
uv run apo-extract run -c configs/hiring_resume.yaml  # needs OpenRouter credits
uv run apo-extract export-web -c configs/swimming.yaml  # frontend snapshot
```

Per-run analysis (full ledger, hypotheses, subtree deltas) is auto-generated in
`runs/<id>/report.md`. Offline checks: 72 tests green (`uv run pytest`).
