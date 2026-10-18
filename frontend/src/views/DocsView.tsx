import { AnatomyDiagram } from "@/components/diagrams/AnatomyDiagram"
import { AttributionDiagram } from "@/components/diagrams/AttributionDiagram"
import { CrashSafetyDiagram } from "@/components/diagrams/CrashSafetyDiagram"
import { DedupDiagram } from "@/components/diagrams/DedupDiagram"
import { GateDiagram } from "@/components/diagrams/GateDiagram"
import { PipelineDiagram } from "@/components/diagrams/PipelineDiagram"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import type { RunSnapshot, SubtreeScore } from "@/lib/types"

function Topic({
  n,
  title,
  lede,
  children,
  i,
}: {
  n: string
  title: string
  lede: string
  children: React.ReactNode
  i: number
}) {
  return (
    <section className="reveal space-y-4" style={{ "--i": i } as React.CSSProperties}>
      <div className="flex items-baseline gap-3">
        <span className="text-redline font-mono text-[12px] font-semibold">{n}</span>
        <h2 className="font-serif text-[26px] font-semibold tracking-tight">{title}</h2>
      </div>
      <div className="rule h-px bg-foreground/50" />
      <p className="text-muted-foreground max-w-3xl text-[13.5px] leading-relaxed">{lede}</p>
      {children}
    </section>
  )
}

function Chips({ items }: { items: Array<[string, string]> }) {
  return (
    <div className="flex flex-wrap gap-2">
      {items.map(([k, v]) => (
        <Badge key={k} variant="outline" className="gap-1.5 font-mono text-[10.5px] font-normal">
          <span className="text-muted-foreground">{k}</span>
          <span className="font-semibold">{v}</span>
        </Badge>
      ))}
    </div>
  )
}

export function DocsView({ run }: { run: RunSnapshot | null }) {
  const opt = run?.config?.optimizer ?? {}
  const rootDollar: SubtreeScore | undefined = run?.root_val?.merged_subtrees?.["$"]
  const nDirectives = run ? Object.keys(run.seed.artifact.subtree_directives).length : 10

  return (
    <div className="space-y-14">
      <Topic
        i={0}
        n="§1"
        title="A loop that only keeps what it can prove"
        lede="The optimizer treats the prompt as four layered pieces and mutates exactly one field-scoped directive at a time. Every candidate is first screened on the train split — the cheapest signal — and only survivors are allowed to spend validation budget. Selection is a banded Pareto front, so a candidate that trades a hair of val F1 for macro-level consistency is not thrown away."
      >
        <PipelineDiagram />
        <div className="grid gap-3 md:grid-cols-3">
          {[
            ["diagnose", "validation errors are clustered per $.field; the worst subtree is attacked first — never the whole prompt at once"],
            ["propose", "the mutator writes one directive from schema + error evidence; template strategies cover hardening, crossover, and checklist edits"],
            ["prove", "stage-1 gate on train (cheap), stage-2 on val; archived candidates must not be dominated on (val F1, aggregate) within tolerance"],
          ].map(([k, v], idx) => (
            <Card key={k} className="py-4" style={{ "--i": idx } as React.CSSProperties}>
              <CardContent className="px-4">
                <div className="kicker mb-1.5 text-redline">{k}</div>
                <p className="text-[12.5px] leading-relaxed text-muted-foreground">{v}</p>
              </CardContent>
            </Card>
          ))}
        </div>
      </Topic>

      <Topic
        i={1}
        n="§2"
        title="The artifact: four layers, one of them alive"
        lede="A prompt artifact is not one blob of text. The preamble frames the task, the null policy pins the output contract, the verification checklist appends a self-check, and the subtree directives carry per-field guidance. Only the last two layers are writable after seeding — everything else is frozen so that gains are attributable."
      >
        <AnatomyDiagram nDirectives={nDirectives} />
        <p className="spec-note max-w-3xl">
          invariant — a directive is addressed by JSON path ($.experience[0].company), so a rollback can strip one
          subtree&apos;s guidance without touching its siblings; the rendered prompt always stays a single call unless the
          field count crosses the chunking threshold
        </p>
      </Topic>

      <Topic
        i={2}
        n="§3"
        title="Scoring: every error has an owner"
        lede="The walker descends the schema tree leaf by leaf and asks the cheapest metric that can decide: exact match for strings, tolerance for numbers, embedding-free semantic checks or an LLM judge for paraphrases, Hungarian matching for arrays. Each leaf ends up in exactly one bucket — and true negatives are deliberately not counted, so padding a document with nulls can never buy score."
      >
        <AttributionDiagram />
        {rootDollar ? (
          <Chips
            items={[
              ["root TP", rootDollar.tp.toFixed(1)],
              ["root FP", rootDollar.fp.toFixed(1)],
              ["root FN", rootDollar.fn.toFixed(1)],
              ["micro F1", rootDollar.f1.toFixed(4)],
              ["json valid", run?.root_val ? `${Math.round(run.root_val.json_valid_rate * 100)}%` : "—"],
            ]}
          />
        ) : null}
      </Topic>

      <Topic
        i={3}
        n="§4"
        title="Gates, bands, and the held-out test"
        lede="A candidate dies at the first gate it fails. Train regression beyond ε is rejected before any val spend; on val, it must either beat the incumbent or add a non-dominated point to the archive. The archive keeps top-k candidates per 0.02 val band, which prevents the beam collapsing onto one lucky artifact. The test split is scored exactly once, when the loop is over."
      >
        <GateDiagram />
        <Chips
          items={[
            ["train gate ε", String(opt.stage1_max_train_regression ?? "—")],
            ["min val improvement", String(opt.min_val_improvement ?? "—")],
            ["pareto tolerance", String(opt.pareto_aggregate_tolerance ?? "—")],
            ["beam width", String(opt.beam_width ?? "—")],
            ["stall patience", String(opt.stall_patience ?? "—")],
          ]}
        />
      </Topic>

      <Topic
        i={4}
        n="§5"
        title="No re-proposing the same idea twice"
        lede="Before a candidate is evaluated, its delta text is compared against every sibling with the same target via Jaccard similarity on normalized tokens. Above the threshold it is skipped as a duplicate. Failed hypotheses go on a tabu board so the mutator never retries them verbatim — this is what keeps a 12-iteration budget from being burned on one idea wearing different hats."
      >
        <DedupDiagram />
        <p className="spec-note max-w-3xl">
          threshold — similarity {String(opt.duplicate_similarity ?? 0.92)} on (target subtree × delta text); tabu
          entries are scoped to the run, not global
        </p>
      </Topic>

      <Topic
        i={5}
        n="§6"
        title="Crash-safe by construction"
        lede="Every completed LLM call is committed before the next one starts: predictions keyed by (prompt hash × document × model), judge verdicts keyed by content, iterations in SQLite WAL plus raw request/response bodies on disk. Killing the process — or hitting a quota wall — loses at most one in-flight call. Resume replays from the caches and picks up at the first un-evaluated candidate."
      >
        <CrashSafetyDiagram />
        {run ? (
          <Chips
            items={[
              ["cached predictions", String(run.n_cached_predictions)],
              ["logged calls", String(run.llm_stats.reduce((a, s) => a + s.n, 0))],
              ["status", run.run.status],
              ["since export", new Date(run.exported_at * 1000).toISOString().slice(0, 16).replace("T", " ") + " UTC"],
            ]}
          />
        ) : null}
      </Topic>
    </div>
  )
}
