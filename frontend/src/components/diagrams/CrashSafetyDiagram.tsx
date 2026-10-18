import { Fig } from "./Fig"

function Layer({
  y,
  title,
  sub,
  keys,
}: {
  y: number
  title: string
  sub: string
  keys: string
}) {
  return (
    <g>
      <rect x={150} y={y} width={620} height={62} rx={7} className="fill-card stroke-foreground/70" strokeWidth={1.3} />
      <rect x={150} y={y} width={8} height={62} rx={4} className="fill-primary" />
      <text x={178} y={y + 25} className="fill-foreground font-mono text-[11.5px] font-semibold">
        {title}
      </text>
      <text x={178} y={y + 44} className="fill-muted-foreground font-mono text-[9.5px]">
        {sub}
      </text>
      <text x={756} y={y + 36} textAnchor="end" className="fill-bluepen font-mono text-[9.5px]">
        {keys}
      </text>
    </g>
  )
}

export function CrashSafetyDiagram() {
  return (
    <Fig
      title="Fig. 6 — crash-safety layers"
      note="kill -9 at any moment loses at most one in-flight call"
      viewBox="0 0 1160 340"
    >
      <defs>
        <marker id="crash-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" className="fill-moss" />
        </marker>
      </defs>

      <Layer
        y={36}
        title="PREDICTIONS CACHE — sqlite"
        sub="extractor output per artifact × document"
        keys="key: prompt_hash × pdf_hash × model"
      />
      <Layer
        y={122}
        title="SCORING CACHE — sqlite"
        sub="judge verdicts per (artifact, doc, field)"
        keys="repeat judges cost zero tokens"
      />
      <Layer
        y={208}
        title="ITERATION COMMITS — wal + per-iter dirs"
        sub="state.db · llm_calls.jsonl · raw request/response bodies"
        keys="committed before the next call starts"
      />

      <path d="M 96 36 L 96 270" className="stroke-moss/70" strokeWidth={1.6} strokeDasharray="6 5" />
      <path d="M 96 270 C 40 270, 40 60, 96 44" fill="none" className="stroke-moss" strokeWidth={1.6} markerEnd="url(#crash-arrow)" />
      <text
        x={22}
        y={160}
        className="fill-moss font-serif text-[13px] italic"
        transform="rotate(-90 22 160)"
        textAnchor="middle"
      >
        resume replays from the layers
      </text>

      <line x1={1120} y1={20} x2={1120} y2={320} className="stroke-redline/60" strokeWidth={1.8} strokeDasharray="8 6" />
      <text x={1112} y={40} textAnchor="end" className="fill-redline font-serif text-[14px] italic">
        process dies here
      </text>
      <text x={1112} y={60} textAnchor="end" className="fill-redline font-serif text-[14px] italic">
        (ctrl-c, quota, laptop lid)
      </text>

      <text x={150} y={300} className="fill-muted-foreground font-serif text-[12.5px] italic">
        resume re-uses cached predictions and judge verdicts — the optimizer picks up at the first
      </text>
      <text x={150} y={320} className="fill-muted-foreground font-serif text-[12.5px] italic">
        un-evaluated candidate; config may drift, caches are keyed by content, not by config hash
      </text>
    </Fig>
  )
}
