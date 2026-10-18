import { IterationCw, PenLine, Sparkles } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import type { RunSnapshot } from "@/lib/types"

function DirectiveDiff({
  seed,
  best,
}: {
  seed: Record<string, string>
  best: Record<string, string>
}) {
  const changed = Object.keys(best).filter(
    (k) => (best[k] ?? "").trim() !== (seed[k] ?? "").trim()
  )
  if (!changed.length) {
    return (
      <p className="text-muted-foreground font-mono text-[11px]">
        no accepted mutations yet — best is the root artifact
      </p>
    )
  }
  return (
    <div className="space-y-2">
      {changed.slice(0, 4).map((k) => (
        <div key={k} className="rounded-md border border-redline/40 bg-redline/5 px-3 py-2">
          <div className="flex items-center justify-between gap-2">
            <span className="font-mono text-[10.5px] font-semibold text-redline">{k}</span>
            <PenLine className="size-3 text-redline" />
          </div>
          <p className="mt-1 line-clamp-2 font-mono text-[10.5px] leading-relaxed">
            {(best[k] ?? "").trim() || "—"}
          </p>
        </div>
      ))}
      {changed.length > 4 ? (
        <p className="text-muted-foreground font-mono text-[10.5px]">
          +{changed.length - 4} more rewritten directives
        </p>
      ) : null}
    </div>
  )
}

export function PromptIO({ run }: { run: RunSnapshot }) {
  const seed = run.seed
  const best = run.best
  const nIter = run.iterations.length
  const accepted = run.iterations.filter((i) => i.decision === "ACCEPTED").length

  return (
    <div className="grid items-stretch gap-4 lg:grid-cols-[1fr_auto_1fr]">
      <Card className="reveal" style={{ "--i": 0 } as React.CSSProperties}>
        <CardHeader className="border-b">
          <div className="flex items-center justify-between">
            <span className="kicker">01 · input</span>
            <Badge variant="outline" className="font-mono text-[10px]">
              seed artifact
            </Badge>
          </div>
          <CardTitle className="font-serif text-xl">The prompt you started with</CardTitle>
          <CardDescription className="font-mono text-[11px]">
            frozen preamble + empty $.directives · {Object.keys(seed.artifact.subtree_directives).length} fields scoped
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="prompt-scroll">{seed.text}</div>
        </CardContent>
      </Card>

      <div className="flex flex-row items-center justify-center gap-2 lg:flex-col lg:px-2">
        <Separator orientation="vertical" className="hidden h-full lg:block lg:flex-1" />
        <div className="flex flex-col items-center gap-2 py-2">
          <div className="flex size-11 items-center justify-center rounded-full border-2 border-primary bg-card shadow-sm">
            <IterationCw className="size-5 text-primary" />
          </div>
          <span className="kicker whitespace-nowrap">SR-PBO</span>
          <Badge variant="secondary" className="font-mono text-[10px]">
            {nIter} iter · {accepted} kept
          </Badge>
        </div>
        <Separator orientation="vertical" className="hidden h-full lg:block lg:flex-1" />
      </div>

      <Card className="reveal border-primary/40" style={{ "--i": 1 } as React.CSSProperties}>
        <CardHeader className="border-b">
          <div className="flex items-center justify-between">
            <span className="kicker">02 · output</span>
            <Badge className="gap-1 font-mono text-[10px]">
              <Sparkles className="size-3" />
              {best.candidate_id}
            </Badge>
          </div>
          <CardTitle className="font-serif text-xl">The prompt the loop earned</CardTitle>
          <CardDescription className="font-mono text-[11px]">
            generation {best.artifact.generation} · every accepted delta traceable to an iteration
          </CardDescription>
        </CardHeader>
        <CardContent className="flex h-full flex-col gap-4">
          <div className="prompt-scroll flex-1">{best.text}</div>
          <DirectiveDiff seed={seed.artifact.subtree_directives} best={best.artifact.subtree_directives} />
        </CardContent>
      </Card>
    </div>
  )
}
