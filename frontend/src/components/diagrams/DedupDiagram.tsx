import { Fig } from "./Fig"

function DirectiveCard({
  x,
  y,
  label,
  lines,
  accent,
}: {
  x: number
  y: number
  label: string
  lines: string[]
  accent: string
}) {
  return (
    <g>
      <rect x={x} y={y} width={330} height={104} rx={8} className="fill-card stroke-foreground/70" strokeWidth={1.3} />
      <text x={x + 16} y={y + 24} className={`font-mono text-[10.5px] font-semibold ${accent}`}>
        {label}
      </text>
      {lines.map((line, i) => (
        <text key={i} x={x + 16} y={y + 46 + i * 17} className="fill-foreground font-mono text-[10px]">
          {line}
        </text>
      ))}
    </g>
  )
}

export function DedupDiagram() {
  return (
    <Fig
      title="Fig. 5 — delta dedup & the tabu board"
      note="similarity is computed per (target subtree, delta text)"
      viewBox="0 0 1160 300"
    >
      <DirectiveCard
        x={40}
        y={64}
        label="CANDIDATE A — iter 4 · $.skills"
        lines={["extract each tool named in the doc;", "keep original casing; no duplicates"]}
        accent="fill-bluepen"
      />
      <DirectiveCard
        x={230}
        y={186}
        label="CANDIDATE B — iter 7 · $.skills"
        lines={["keep tools in original casing; never", "extract the same tool twice"]}
        accent="fill-bluepen"
      />

      <rect x={380} y={150} width={130} height={34} className="fill-redline/12 stroke-redline/40" strokeWidth={1} rx={4} />
      <text x={445} y={171} textAnchor="middle" className="fill-redline font-mono text-[9.5px]">
        J(A,B) = 0.94
      </text>

      <text x={600} y={104} className="fill-redline font-serif text-[14px] italic">
        Jaccard(A, B) = |A∩B| / |A∪B| ≥ 0.92 →
      </text>
      <text x={600} y={126} className="fill-redline font-serif text-[14px] italic">
        same idea, same target — sibling is skipped
      </text>
      <text x={600} y={162} className="fill-muted-foreground font-serif text-[12.5px] italic">
        stops the loop from re-proposing &quot;original casing&quot; forever
      </text>

      <DirectiveCard
        x={600}
        y={186}
        label="CANDIDATE C — iter 8 · $.skills"
        lines={["map synonyms to the schema enum;", "report unknown tools as null"]}
        accent="fill-moss"
      />
      <text x={946} y={240} className="fill-moss font-mono text-[10px]">
        novel → admitted
      </text>

      <rect x={946} y={64} width={174} height={104} rx={8} className="fill-card stroke-foreground/70" strokeWidth={1.3} />
      <text x={962} y={88} className="fill-foreground font-mono text-[10px] font-semibold uppercase tracking-[0.12em]">
        tabu board
      </text>
      {["iter 4 · $.email — prefer work email", "iter 6 · $.role — title case only"].map((t, i) => (
        <g key={i}>
          <text x={962} y={108 + i * 20} className="fill-muted-foreground font-mono text-[9px]">
            {t}
          </text>
          <line x1={962} y1={104 + i * 20} x2={1108} y2={104 + i * 20} className="stroke-redline/70" strokeWidth={1.1} />
        </g>
      ))}
      <text x={962} y={158} className="fill-muted-foreground font-serif text-[10.5px] italic">
        failed ideas are not retried
      </text>
    </Fig>
  )
}
