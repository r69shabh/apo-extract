import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { fmt, fmtInt, type RunSnapshot } from "@/lib/types"

function Delta({ root, best }: { root: number | null | undefined; best: number | null | undefined }) {
  if (root == null || best == null) return null
  const d = best - root
  const Icon = d > 0.0005 ? ArrowUpRight : d < -0.0005 ? ArrowDownRight : Minus
  return (
    <span
      className={
        d > 0.0005
          ? "text-moss inline-flex items-center gap-0.5 font-mono text-[11px] font-semibold"
          : d < -0.0005
            ? "text-redline inline-flex items-center gap-0.5 font-mono text-[11px] font-semibold"
            : "text-muted-foreground inline-flex items-center gap-0.5 font-mono text-[11px]"
      }
    >
      <Icon className="size-3" />
      {d > 0 ? "+" : ""}
      {d.toFixed(4)}
    </span>
  )
}

function Kpi({
  label,
  value,
  sub,
  children,
  i,
}: {
  label: string
  value: string
  sub?: string
  children?: React.ReactNode
  i: number
}) {
  return (
    <Card className="reveal gap-3 py-4" style={{ "--i": i } as React.CSSProperties}>
      <CardHeader className="px-4">
        <CardTitle className="kicker font-normal">{label}</CardTitle>
      </CardHeader>
      <CardContent className="px-4">
        <div className="flex items-baseline gap-2">
          <span className="font-serif text-[26px] leading-none font-semibold tabular-nums">{value}</span>
          {children}
        </div>
        {sub ? <p className="text-muted-foreground mt-1.5 font-mono text-[10.5px]">{sub}</p> : null}
      </CardContent>
    </Card>
  )
}

export function StatsRow({ run }: { run: RunSnapshot }) {
  const rootF1 = run.root_val?.f1 ?? null
  const rootAgg = run.root_val?.aggregate_f1 ?? null
  const bestF1 = run.summary?.val_f1 ?? run.archive[0]?.val_f1 ?? rootF1
  const tokens = run.llm_stats.reduce((a, s) => a + (s.it ?? 0) + (s.ot ?? 0), 0)
  const calls = run.llm_stats.reduce((a, s) => a + (s.n ?? 0), 0)
  const usd = run.run.budget_spent_usd ?? 0

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      <Kpi i={0} label="val F1 · held-in" value={fmt(bestF1)} sub="best candidate, selection split">
        <Delta root={rootF1} best={bestF1} />
      </Kpi>
      <Kpi i={1} label="aggregate · macro subtree" value={fmt(run.archive[0]?.aggregate_f1 ?? rootAgg)} sub="honest per-subtree view, no beam-size bias">
        <Delta root={rootAgg} best={run.archive[0]?.aggregate_f1 ?? rootAgg} />
      </Kpi>
      <Kpi i={2} label="test F1 · held-out" value={fmt(run.summary?.test_f1)} sub="scored once, after the loop ends">
        <Delta root={run.summary?.seed_test_f1} best={run.summary?.test_f1} />
      </Kpi>
      <Kpi
        i={3}
        label="spend"
        value={usd > 0 ? `$${usd.toFixed(2)}` : "0.00"}
        sub={`${fmtInt(tokens)} tokens · ${fmtInt(calls)} logged calls · free tier`}
      >
        <Badge variant="outline" className="font-mono text-[10px]">
          {run.n_cached_predictions} cached preds
        </Badge>
      </Kpi>
    </div>
  )
}
