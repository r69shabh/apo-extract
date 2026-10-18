import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { RunSnapshot } from "@/lib/types"

function Row({ label, value, mono = true }: { label: string; value: React.ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-dashed py-1.5 last:border-0">
      <span className="kicker">{label}</span>
      <span className={mono ? "font-mono text-[11px] text-right" : "text-[12px] text-right"}>{value}</span>
    </div>
  )
}

const SPLIT_STYLE: Record<string, string> = {
  train: "border-primary/40 bg-primary/5",
  val: "border-bluepen/40 bg-bluepen/5 text-bluepen",
  test: "border-moss/50 bg-moss/10 text-moss",
}

export function RunConfig({ run }: { run: RunSnapshot }) {
  const cfg = run.config
  const llms = cfg?.llms ?? {}
  const budget = cfg?.budget ?? {}
  const opt = cfg?.optimizer ?? {}
  const scoring = cfg?.scoring ?? {}
  const split = run.dataset.split
  const wallH = budget.max_wall_clock_seconds ? (budget.max_wall_clock_seconds / 3600).toFixed(1) : "—"

  return (
    <Card className="reveal" style={{ "--i": 4 } as React.CSSProperties}>
      <CardHeader>
        <CardTitle className="font-serif text-lg">Run dossier</CardTitle>
        <CardDescription className="font-mono text-[11px]">
          inputs, budget, and the split the loop never crosses
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-8 lg:grid-cols-3">
        <div>
          <h4 className="kicker mb-2">models</h4>
          <Row label="extractor" value={llms.extractor?.model ?? "—"} />
          <Row label="mutator" value={llms.optimizer_mutator?.model ?? "—"} />
          <Row label="judge" value={(scoring.judge_model ?? llms.scoring_judge?.model ?? "—").replace(/^gemini\//, "")} />
          <Row label="provider" value={llms.extractor?.provider ?? "—"} />
          <Row label="extraction" value={opt.extraction_mode ?? "—"} />
        </div>
        <div>
          <h4 className="kicker mb-2">budget & loop</h4>
          <Row label="max iters" value={budget.max_iterations ?? "—"} />
          <Row label="wall clock" value={`${wallH} h`} />
          <Row label="cost cap" value={`$${budget.max_cost_usd ?? "—"}`} />
          <Row label="beam width" value={opt.beam_width ?? "—"} />
          <Row label="dup similarity" value={opt.duplicate_similarity ?? "—"} />
          <Row label="stall patience" value={opt.stall_patience ?? "—"} />
        </div>
        <div>
          <h4 className="kicker mb-2">dataset · {run.dataset.name ?? run.dataset.path}</h4>
          <Row label="documents" value={run.dataset.n_documents} />
          <Row label="schema" value={`${run.dataset.schema_fields.length} top-level fields`} />
          <div className="mt-3 space-y-2.5">
            {(["train", "val", "test"] as const).map((name) => (
              <div key={name}>
                <div className="mb-1 flex items-center gap-2">
                  <Badge variant="outline" className={`font-mono text-[10px] ${SPLIT_STYLE[name]}`}>
                    {name}
                  </Badge>
                  <span className="text-muted-foreground font-mono text-[10px]">{split[name].length} docs</span>
                </div>
                <div className="flex flex-wrap gap-1">
                  {split[name].map((doc) => (
                    <span key={doc} className="bg-muted rounded px-1.5 py-0.5 font-mono text-[9.5px] text-muted-foreground">
                      {doc}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
