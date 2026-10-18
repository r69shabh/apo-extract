import { Fig } from "./Fig"

function Node({
  x,
  y = 92,
  n,
  title,
  sub,
}: {
  x: number
  y?: number
  n: string
  title: string
  sub: string
}) {
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={124}
        height={66}
        rx={7}
        className="fill-card stroke-foreground/80"
        strokeWidth={1.3}
      />
      <rect x={x + 9} y={y + 9} width={24} height={24} rx={5} className="fill-primary" />
      <text
        x={x + 21}
        y={y + 26}
        textAnchor="middle"
        className="fill-primary-foreground font-mono text-[11px] font-semibold"
      >
        {n}
      </text>
      <text x={x + 42} y={y + 26} className="fill-foreground font-mono text-[11px] font-semibold">
        {title}
      </text>
      <text x={x + 42} y={y + 45} className="fill-muted-foreground font-mono text-[9px]">
        {sub}
      </text>
    </g>
  )
}

function Wire({ x1, x2, y = 125, tag, d = 0 }: { x1: number; x2: number; y?: number; tag: string; d?: number }) {
  return (
    <g>
      <line
        x1={x1}
        y1={y}
        x2={x2}
        y2={y}
        className="wire stroke-foreground/70"
        strokeWidth={1.4}
        markerEnd="url(#pipe-arrow)"
        style={{ "--d": d } as React.CSSProperties}
      />
      <text
        x={(x1 + x2) / 2}
        y={y - 8}
        textAnchor="middle"
        className="fill-muted-foreground font-mono text-[8px] uppercase"
      >
        {tag}
      </text>
    </g>
  )
}

function Wavy({ x, y, width = 148 }: { x: number; y: number; width?: number }) {
  const step = 12
  const n = Math.floor(width / step)
  const d = `M ${x} ${y} ` + Array.from({ length: n }, (_, i) => `q ${step / 2} -5 ${step} 0`).join(" ")
  return <path d={d} className="stroke-redline/70" strokeWidth={1.6} fill="none" />
}

export function PipelineDiagram() {
  return (
    <Fig title="Fig. 1 — SR-PBO optimization loop" note="one iteration ≈ 3 train + 2 val doc evaluations" viewBox="0 0 1160 336">
      <defs>
        <marker id="pipe-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" className="fill-foreground/70" />
        </marker>
        <marker id="pipe-arrow-amber" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" className="fill-amberink" />
        </marker>
      </defs>

      <Node x={22} n="01" title="SEED" sub="seed + schema" />
      <Node x={182} n="02" title="BASELINE" sub="root, train" />
      <Node x={342} n="03" title="DIAGNOSE" sub="worst first" />
      <Node x={502} n="04" title="PROPOSE" sub="one directive" />
      <Node x={662} n="05" title="GATES" sub="train → val" />
      <Node x={822} n="06" title="FRONT" sub="banded front" />
      <Node x={982} n="07" title="TEST" sub="held-out, once" />

      <Wire x1={146} x2={182} tag="init" d={0} />
      <Wire x1={306} x2={342} tag="eval" d={1} />
      <Wire x1={466} x2={502} tag="rank" d={2} />
      <Wire x1={626} x2={662} tag="draft" d={3} />
      <Wire x1={786} x2={822} tag="accept" d={4} />
      <Wire x1={946} x2={982} tag="final" d={5} />

      <path
        d="M 884 158 C 884 226, 564 226, 564 158"
        fill="none"
        className="stroke-amberink/80"
        strokeWidth={1.4}
        strokeDasharray="5 5"
        markerEnd="url(#pipe-arrow-amber)"
      />
      <text x={724} y={242} textAnchor="middle" className="fill-amberink font-serif text-[13px] italic">
        subtree rollback pass — beam keeps top-k per band
      </text>

      <text x={724} y={196} textAnchor="middle" className="fill-redline font-serif text-[13px] italic">
        reject if train &lt; root − ε
      </text>
      <Wavy x={650} y={203} width={148} />

      <text x={940} y={218} textAnchor="middle" className="fill-redline font-serif text-[13px] italic">
        ranked by val F1 — test scored once, never reused
      </text>
      <Wavy x={830} y={225} width={220} />

      <line x1={404} y1={158} x2={404} y2={262} className="stroke-bluepen/60" strokeWidth={1.1} strokeDasharray="3 4" />
      <text x={404} y={276} textAnchor="middle" className="fill-bluepen font-serif text-[12.5px] italic">
        mutator sees schema + parse errors only
      </text>

      <line x1={22} y1={305} x2={1106} y2={305} className="stroke-foreground/50" strokeWidth={1.1} />
      {[22, 342, 502, 662, 822, 982, 1106].map((x) => (
        <line key={x} x1={x} y1={299} x2={x} y2={311} className="stroke-foreground/50" strokeWidth={1.1} />
      ))}
      <rect x={502} y={292} width={444} height={26} rx={4} fill="none" className="stroke-redline/60" strokeWidth={1.2} strokeDasharray="4 4" />
      <text x={724} y={326} textAnchor="middle" className="fill-redline font-mono text-[9.5px] uppercase tracking-[0.18em]">
        one iteration — max 12, stall patience 3
      </text>
    </Fig>
  )
}
