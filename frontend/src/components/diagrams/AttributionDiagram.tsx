import { Fig } from "./Fig"

type Outcome = "tp" | "fn" | "fp" | "partial" | "tn"

const OUTCOME_META: Record<Outcome, { glyph: string; color: string; label: string }> = {
  tp: { glyph: "=", color: "text-moss", label: "TP — matched" },
  fn: { glyph: "−", color: "text-redline", label: "FN — gold missed" },
  fp: { glyph: "+", color: "text-redline", label: "FP — extra value" },
  partial: { glyph: "~", color: "text-bluepen", label: "soft score 0<f1<1" },
  tn: { glyph: "∅", color: "text-muted-foreground", label: "TN — both null, excluded" },
}

const ROWS: Array<{ field: string; gold: string; pred: string; outcome: Outcome }> = [
  { field: "$.name", gold: '"Ada Reyes"', pred: '"Ada Reyes"', outcome: "tp" },
  { field: "$.role", gold: '"ML Engineer"', pred: '"ML Engineer"', outcome: "tp" },
  { field: "$.email", gold: '"ada@lab.io"', pred: "null", outcome: "fn" },
  { field: "$.skills[]", gold: '["python", "sql"]', pred: '["python", "sql", "go"]', outcome: "fp" },
  { field: "$.experience[0].title", gold: '"Senior SWE"', pred: '"Senior Software Eng."', outcome: "partial" },
  { field: "$.education[3]", gold: "null", pred: "null", outcome: "tn" },
]

export function AttributionDiagram() {
  const rowY = (i: number) => 96 + i * 38

  return (
    <Fig
      title="Fig. 3 — error attribution, field by field"
      note="schema-tree walk · per-leaf comparison"
      viewBox="0 0 1160 420"
    >
      <defs>
        <marker id="attr-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" className="fill-foreground/60" />
        </marker>
      </defs>

      <rect x={36} y={54} width={380} height={300} rx={8} className="fill-card stroke-foreground/70" strokeWidth={1.3} />
      <text x={54} y={78} className="fill-foreground font-mono text-[11px] font-semibold uppercase tracking-[0.16em]">
        gold — ground truth
      </text>
      <line x1={36} y1={88} x2={416} y2={88} className="stroke-foreground/30" strokeWidth={1} />

      <rect x={744} y={54} width={380} height={300} rx={8} className="fill-card stroke-foreground/70" strokeWidth={1.3} />
      <text x={762} y={78} className="fill-foreground font-mono text-[11px] font-semibold uppercase tracking-[0.16em]">
        extracted — model output
      </text>
      <line x1={744} y1={88} x2={1124} y2={88} className="stroke-foreground/30" strokeWidth={1} />

      {ROWS.map((row, i) => {
        const y = rowY(i)
        const meta = OUTCOME_META[row.outcome]
        return (
          <g key={row.field}>
            <text x={54} y={y} className="fill-muted-foreground font-mono text-[9.5px]">
              {row.field}
            </text>
            <text x={54} y={y + 15} className="fill-foreground font-mono text-[11px]">
              {row.gold}
            </text>
            <text x={762} y={y} className="fill-muted-foreground font-mono text-[9.5px]">
              {row.field}
            </text>
            <text
              x={762}
              y={y + 15}
              className={
                row.outcome === "fn"
                  ? "fill-amberink font-mono text-[11px] font-semibold"
                  : row.outcome === "tn"
                    ? "fill-muted-foreground font-mono text-[11px] italic"
                    : "fill-foreground font-mono text-[11px]"
              }
            >
              {row.pred}
            </text>

            {row.outcome !== "tn" && (
              <line
                x1={416}
                y1={y + 8}
                x2={744}
                y2={y + 8}
                className="stroke-foreground/25"
                strokeWidth={1}
                strokeDasharray="2 4"
              />
            )}
            {row.outcome === "tn" && (
              <line x1={416} y1={y + 8} x2={744} y2={y + 8} className="stroke-foreground/15" strokeWidth={1} />
            )}

            <circle cx={580} cy={y + 8} r={11} className={row.outcome === "tn" ? "fill-muted stroke-border" : "fill-card stroke-foreground/50"} strokeWidth={1.2} />
            <text x={580} y={y + 12} textAnchor="middle" className={`font-mono text-[12px] font-semibold ${meta.color}`}>
              {meta.glyph}
            </text>
            {row.outcome === "fp" && (
              <text x={580} y={y + 34} textAnchor="middle" className="fill-redline font-serif text-[11.5px] italic">
                &quot;go&quot; not in gold
              </text>
            )}
            {row.outcome === "fn" && (
              <text x={580} y={y + 34} textAnchor="middle" className="fill-amberink font-serif text-[11.5px] italic">
                lost to null
              </text>
            )}
          </g>
        )
      })}

      <g>
        <rect x={36} y={372} width={540} height={30} rx={6} className="fill-redline/8 stroke-redline/50" strokeWidth={1.1} />
        <text x={54} y={391} className="fill-redline font-mono text-[11px]">
          micro F1 = 2·TP / (2·TP + FP + FN)   ·   TP/FP/FN are fractional — soft leaf scores
        </text>
      </g>
      <text x={762} y={391} className="fill-muted-foreground font-mono text-[11px]">
        subtree F1 = mean over $.fields (macro) · TN never counted
      </text>
    </Fig>
  )
}
