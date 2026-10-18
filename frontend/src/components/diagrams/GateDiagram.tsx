import { Fig } from "./Fig"

export function GateDiagram() {
  return (
    <Fig
      title="Fig. 4 — staged gates & the pareto archive"
      note="cheap gate first — train is never trusted alone"
      viewBox="0 0 1160 300"
    >
      <defs>
        <marker id="gate-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" className="fill-foreground/70" />
        </marker>
      </defs>

      <text x={36} y={62} className="fill-muted-foreground font-mono text-[9.5px] uppercase tracking-[0.16em]">
        candidate flow
      </text>

      <line x1={36} y1={120} x2={228} y2={120} className="stroke-foreground/60" strokeWidth={1.3} markerEnd="url(#gate-arrow)" />

      <rect x={236} y={62} width={10} height={116} rx={3} className="fill-primary" />
      <text x={241} y={40} textAnchor="middle" className="fill-foreground font-mono text-[10px] font-semibold">
        GATE 1 · TRAIN
      </text>
      <text x={241} y={56} textAnchor="middle" className="fill-muted-foreground font-mono text-[9px]">
        train_F1 ≥ root − ε
      </text>

      <line x1={246} y1={120} x2={432} y2={120} className="stroke-foreground/60" strokeWidth={1.3} markerEnd="url(#gate-arrow)" />

      <rect x={440} y={62} width={10} height={116} rx={3} className="fill-primary" />
      <text x={445} y={40} textAnchor="middle" className="fill-foreground font-mono text-[10px] font-semibold">
        GATE 2 · VAL
      </text>
      <text x={445} y={56} textAnchor="middle" className="fill-muted-foreground font-mono text-[9px]">
        val gain or pareto-add
      </text>

      <line x1={450} y1={120} x2={640} y2={120} className="stroke-foreground/60" strokeWidth={1.3} markerEnd="url(#gate-arrow)" />

      {[0, 1, 2].map((k) => (
        <rect key={k} x={652 + k * 5} y={96 + k * 6} width={64} height={40} rx={5} className="fill-card stroke-foreground/70" strokeWidth={1.2} />
      ))}
      <text x={690} y={170} textAnchor="middle" className="fill-foreground font-mono text-[10px] font-semibold">
        ARCHIVE
      </text>
      <text x={690} y={184} textAnchor="middle" className="fill-muted-foreground font-mono text-[9px]">
        banded pareto front
      </text>

      <path d="M 241 178 C 238 214, 205 236, 152 240" fill="none" className="stroke-redline" strokeWidth={1.4} markerEnd="url(#gate-arrow)" />
      <text x={40} y={260} className="fill-redline font-serif text-[12.5px] italic">
        train regressed — dropped before any val spend
      </text>

      <path d="M 445 178 C 450 214, 505 236, 556 240" fill="none" className="stroke-amberink" strokeWidth={1.4} markerEnd="url(#gate-arrow)" />
      <text x={556} y={260} className="fill-amberink font-serif text-[12.5px] italic">
        no val movement — logged, not archived
      </text>

      <line x1={780} y1={40} x2={780} y2={212} className="stroke-foreground/60" strokeWidth={1.2} />
      {Array.from({ length: 8 }).map((_, i) => (
        <g key={i}>
          <rect x={781} y={40 + i * 21.5} width={300} height={21.5} className={i % 2 ? "fill-muted/70" : "fill-muted/30"} />
          <text x={768} y={55 + i * 21.5} textAnchor="end" className="fill-muted-foreground font-mono text-[8.5px]">
            {(0.76 - i * 0.02).toFixed(2)}
          </text>
        </g>
      ))}
      <line x1={780} y1={212} x2={1081} y2={212} className="stroke-foreground/60" strokeWidth={1.2} />
      <text x={930} y={234} textAnchor="middle" className="fill-muted-foreground font-mono text-[9.5px]">
        val F1 discretized into 0.02 bands
      </text>

      <circle cx={880} cy={126} r={6} className="fill-primary" />
      <text x={892} y={122} className="fill-foreground font-mono text-[9.5px]">
        incumbent
      </text>
      <circle cx={990} cy={104} r={6} className="fill-moss" />
      <text x={1002} y={100} className="fill-moss font-mono text-[9.5px]">
        candidate lands same band → dominance check
      </text>
      <circle cx={930} cy={190} r={6} className="fill-card stroke-redline" strokeWidth={1.6} />
      <text x={942} y={194} className="fill-redline font-mono text-[9.5px]">
        dominated — kept out of beam
      </text>

      <text x={36} y={278} className="fill-bluepen font-serif text-[12.5px] italic">
        rollback runs the reverse direction: worst val subtrees are rebuilt from their parent&apos;s directive
      </text>
    </Fig>
  )
}
