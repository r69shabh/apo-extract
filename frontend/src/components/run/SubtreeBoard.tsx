import { useMemo, useState } from "react"
import { Minus, TrendingDown, TrendingUp } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import type { RunSnapshot, SubtreeScore } from "@/lib/types"

const f1Of = (s: SubtreeScore | undefined) =>
  s ? (s.f1 ?? (s as any).macro_subtree_f1 ?? 0) : undefined

export function SubtreeBoard({ run }: { run: RunSnapshot }) {
  const [query, setQuery] = useState("")

  const rootMap = run.root_val?.merged_subtrees ?? {}

  const bestMap = useMemo(() => {
    if (run.archive.length) return run.archive[0].subtree
    for (const iter of [...run.iterations].reverse()) {
      if (iter.subtree_scores_json && iter.decision === "ACCEPTED") {
        try {
          return JSON.parse(iter.subtree_scores_json) as Record<string, SubtreeScore>
        } catch {
          return {}
        }
      }
    }
    return {}
  }, [run])

  const hasBest = Object.keys(bestMap).length > 0

  const rows = useMemo(() => {
    const keys = Array.from(new Set([...Object.keys(rootMap), ...Object.keys(bestMap)])).sort()
    return keys
      .map((k) => {
        const rootF1 = f1Of(rootMap[k])
        const bestF1 = f1Of(bestMap[k])
        const delta = rootF1 !== undefined && bestF1 !== undefined ? bestF1 - rootF1 : undefined
        return { key: k, rootF1, bestF1, delta }
      })
      .sort((a, b) =>
        a.delta !== undefined && b.delta !== undefined
          ? b.delta - a.delta
          : (a.rootF1 ?? Infinity) - (b.rootF1 ?? Infinity)
      )
      .filter((r) => (query ? r.key.toLowerCase().includes(query.toLowerCase()) : true))
  }, [rootMap, bestMap, query])

  const improved = rows.filter((r) => (r.delta ?? 0) > 0.0005).length
  const regressed = rows.filter((r) => (r.delta ?? 0) < -0.0005).length

  return (
    <Card className="reveal" style={{ "--i": 3 } as React.CSSProperties}>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="font-serif text-lg">Subtree scoreboard</CardTitle>
            <CardDescription className="font-mono text-[11px]">
              per-$.field F1 — root vs {run.archive[0]?.candidate_id ?? "best"} · sorted by movement
            </CardDescription>
          </div>
          <div className="flex items-center gap-2">
            {hasBest ? (
              <>
                <Badge variant="outline" className="border-moss/50 bg-moss/10 font-mono text-[10px] text-moss">
                  ▲ {improved} improved
                </Badge>
                <Badge variant="outline" className="border-redline/40 bg-redline/5 font-mono text-[10px] text-redline">
                  ▼ {regressed} regressed
                </Badge>
              </>
            ) : (
              <Badge variant="outline" className="font-mono text-[10px]">
                baseline only — run in progress
              </Badge>
            )}
            <Input
              placeholder="filter $.field…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="h-8 w-44 font-mono text-[11px]"
            />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="max-h-[560px] overflow-auto pr-1">
          <div className="grid grid-cols-[minmax(150px,260px)_1fr_72px_72px_86px] items-center gap-x-3 border-b pb-1.5">
            <span className="kicker">field</span>
            <span className="kicker">
              {hasBest ? "root → best" : "root F1"}
            </span>
            <span className="kicker text-right">root</span>
            <span className="kicker text-right">best</span>
            <span className="kicker text-right">Δ</span>
          </div>
          {rows.length === 0 ? (
            <p className="text-muted-foreground py-6 text-center font-mono text-[11px]">
              no subtree scores yet
            </p>
          ) : (
            rows.map((r, i) => {
              const indicator =
                r.delta === undefined || Math.abs(r.delta) < 0.0005
                  ? "bg-primary/70"
                  : r.delta > 0
                    ? "bg-moss"
                    : "bg-redline"
              return (
                <div
                  key={r.key}
                  className="grid grid-cols-[minmax(150px,260px)_1fr_72px_72px_86px] items-center gap-x-3 border-b border-dashed py-[7px] last:border-0"
                >
                  <span
                    className="truncate font-mono text-[10.5px]"
                    title={r.key}
                  >
                    {r.key}
                  </span>
                  <Progress
                    value={((hasBest ? r.bestF1 : r.rootF1) ?? 0) * 100}
                    indicatorClassName={indicator}
                    className="h-1.5"
                  />
                  <span className="text-right font-mono text-[10.5px] tabular-nums text-muted-foreground">
                    {r.rootF1 !== undefined ? r.rootF1.toFixed(3) : "—"}
                  </span>
                  <span className="text-right font-mono text-[10.5px] tabular-nums">
                    {hasBest ? (r.bestF1 !== undefined ? r.bestF1.toFixed(3) : "—") : "·"}
                  </span>
                  <DeltaCell delta={r.delta} />
                </div>
              )
            })
          )}
        </div>
      </CardContent>
    </Card>
  )
}

function DeltaCell({ delta }: { delta: number | undefined }) {
  if (delta === undefined) return <span className="text-right font-mono text-[10.5px] text-muted-foreground">—</span>
  const Icon = delta > 0.0005 ? TrendingUp : delta < -0.0005 ? TrendingDown : Minus
  return (
    <span
      className={
        "flex items-center justify-end gap-1 font-mono text-[10.5px] tabular-nums " +
        (delta > 0.0005 ? "text-moss" : delta < -0.0005 ? "text-redline" : "text-muted-foreground")
      }
    >
      <Icon className="size-3" />
      {delta > 0 ? "+" : ""}
      {delta.toFixed(3)}
    </span>
  )
}
