import {
  CartesianGrid,
  ComposedChart,
  Line,
  Scatter,
  ScatterChart,
  XAxis,
  YAxis,
} from "recharts"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart"
import type { RunSnapshot } from "@/lib/types"

const trailConfig = {
  best: { label: "best-so-far val F1", color: "var(--moss)" },
  val: { label: "candidate val F1", color: "var(--chart-2)" },
} satisfies ChartConfig

const paretoConfig = {
  archive: { label: "pareto archive", color: "var(--chart-1)" },
  root: { label: "root baseline", color: "var(--chart-2)" },
} satisfies ChartConfig

export function ScoreCharts({ run }: { run: RunSnapshot }) {
  const rootF1 = run.root_val?.f1 ?? 0
  const rootAgg = run.root_val?.aggregate_f1 ?? 0

  const rows = run.iterations.filter((r) => r.val_f1 !== null)
  const trail = rows.map((r) => ({
    x: r.iteration,
    val: r.val_f1,
    train: r.train_f1,
    candidate: r.candidate_id,
  }))
  let running = rootF1
  const bestSoFar = [
    { x: 0, best: rootF1 },
    ...trail.map((p) => {
      running = Math.max(running, p.val ?? 0)
      return { x: p.x, best: running }
    }),
  ]
  const trailData = [{ x: 0, val: rootF1, train: rootF1, candidate: "root" }, ...trail]

  const archived = run.archive.map((e) => ({
    val: e.val_f1,
    aggregate: e.aggregate_f1,
    candidate: e.candidate_id,
  }))
  const rootPoint = [{ val: rootF1, aggregate: rootAgg, candidate: "root" }]

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="reveal" style={{ "--i": 0 } as React.CSSProperties}>
        <CardHeader>
          <CardTitle className="font-serif text-lg">Score trail</CardTitle>
          <CardDescription className="font-mono text-[11px]">
            val F1 per candidate · step line = incumbent · click legend to isolate
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={trailConfig} className="aspect-[16/9] w-full">
            <ComposedChart data={trailData} margin={{ left: 4, right: 12, top: 8 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 6" />
              <XAxis
                dataKey="x"
                type="number"
                domain={[0, (max: number) => Math.max(1, max ?? 0)]}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickFormatter={(v: number) => (v === 0 ? "root" : `i${v}`)}
                allowDecimals={false}
              />
              <YAxis
                domain={[
                  (min: number) => Math.max(0, +(min - 0.04).toFixed(2)),
                  (max: number) => Math.min(1, +(max + 0.04).toFixed(2)),
                ]}
                tickLine={false}
                axisLine={false}
                width={44}
                tickFormatter={(v: number) => v.toFixed(2)}
              />
              <ChartTooltip
                cursor={{ strokeDasharray: "3 3" }}
                content={<ChartTooltipContent labelFormatter={(v) => (v === 0 ? "root baseline" : `iteration ${v}`)} />}
              />
              <ChartLegend content={<ChartLegendContent />} />
              <Scatter
                dataKey="val"
                name="val"
                fill="var(--chart-2)"
                isAnimationActive
              />
              <Line
                data={bestSoFar}
                dataKey="best"
                name="best"
                type="stepAfter"
                stroke="var(--moss)"
                strokeWidth={2}
                dot={false}
                isAnimationActive
              />
            </ComposedChart>
          </ChartContainer>
          {rows.length === 0 ? (
            <p className="text-muted-foreground py-6 text-center font-mono text-[11px]">
              no candidates evaluated yet — root baseline only
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Card className="reveal" style={{ "--i": 1 } as React.CSSProperties}>
        <CardHeader>
          <CardTitle className="font-serif text-lg">Pareto front</CardTitle>
          <CardDescription className="font-mono text-[11px]">
            val F1 × aggregate (macro subtree) · every archived candidate
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={paretoConfig} className="aspect-[16/9] w-full">
            <ScatterChart margin={{ left: 4, right: 16, top: 8 }}>
              <CartesianGrid strokeDasharray="3 6" />
              <XAxis
                dataKey="val"
                type="number"
                domain={["dataMin - 0.03", "dataMax + 0.03"]}
                tickLine={false}
                axisLine={false}
                tickMargin={8}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{ value: "val F1", position: "insideBottom", offset: -4, fontSize: 10 }}
              />
              <YAxis
                dataKey="aggregate"
                type="number"
                domain={["dataMin - 0.03", "dataMax + 0.03"]}
                tickLine={false}
                axisLine={false}
                width={48}
                tickFormatter={(v: number) => v.toFixed(2)}
                label={{ value: "aggregate", angle: -90, position: "insideLeft", fontSize: 10, offset: 8 }}
              />
              <ChartTooltip cursor={{ strokeDasharray: "3 3" }} content={<ChartTooltipContent />} />
              <ChartLegend content={<ChartLegendContent nameKey="value" />} />
              <Scatter data={archived} name="archive" dataKey="aggregate" fill="var(--chart-1)" isAnimationActive />
              <Scatter data={rootPoint} name="root" dataKey="aggregate" fill="var(--chart-2)" isAnimationActive />
            </ScatterChart>
          </ChartContainer>
          <p className="spec-note mt-3">
            note — the beam keeps top-k per 0.02 val band, so near-ties survive instead of one greedy point;
            test split is scored exactly once, after the loop ends
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
