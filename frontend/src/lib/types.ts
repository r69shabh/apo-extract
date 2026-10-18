export interface SubtreeScore {
  tp: number
  fp: number
  fn: number
  precision: number
  recall: number
  f1: number
  count?: number
  mean_score?: number
}

export interface SplitEval {
  f1: number
  aggregate_f1: number
  json_valid_rate: number
  per_doc?: Record<string, unknown>
  merged_subtrees: Record<string, SubtreeScore>
}

export interface IterationRow {
  iteration: number
  candidate_id: string
  parent_id: string | null
  mutation_type: string
  target_subtree: string | null
  hypothesis: string
  train_f1: number | null
  val_f1: number | null
  subtree_scores_json: string | null
  decision: string
  decision_reason: string
}

export interface ArchiveEntry {
  candidate_id: string
  val_f1: number
  aggregate_f1: number
  subtree: Record<string, SubtreeScore>
  created_at: number
}

export interface LlmStat {
  role: string
  model: string
  n: number
  it: number | null
  ot: number | null
  lat_ms: number | null
}

export interface PromptPayload {
  artifact: {
    artifact_id: string
    parent_ids: string[]
    generation: number
    global_preamble: string
    formatting_and_null_policy: string
    subtree_directives: Record<string, string>
    verification_checklist: string
  }
  text: string
}

export interface RunSnapshot {
  run_id: string
  exported_at: number
  run: {
    status: string
    config_path?: string
    budget_spent_tokens?: number
    budget_spent_usd?: number
    current_iteration?: number
    stall?: number
    best_candidate_id?: string | null
    started_at?: number
  }
  config: Record<string, any>
  dataset: {
    name?: string
    path?: string
    n_documents: number
    schema_fields: string[]
    split: { train: string[]; val: string[]; test: string[] }
  }
  seed: PromptPayload
  best: PromptPayload & { candidate_id: string }
  root_val: SplitEval | null
  summary: Record<string, any> | null
  test_per_document: Record<string, any> | null
  iterations: IterationRow[]
  archive: ArchiveEntry[]
  llm_stats: LlmStat[]
  n_cached_predictions: number
}

export const fmt = (x: number | null | undefined, digits = 4) =>
  x === null || x === undefined || Number.isNaN(x) ? "—" : x.toFixed(digits)

export const fmtInt = (x: number | null | undefined) =>
  x === null || x === undefined ? "—" : Math.round(x).toLocaleString()

export const timeAgo = (ts: number) => {
  const s = Math.max(0, Date.now() / 1000 - ts)
  if (s < 60) return "just now"
  if (s < 3600) return `${Math.floor(s / 60)}m ago`
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`
  return `${Math.floor(s / 86400)}d ago`
}
