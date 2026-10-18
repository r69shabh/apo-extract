import { Fig } from "./Fig"

function Slab({
  i,
  y,
  title,
  sub,
  mutable = false,
  children,
}: {
  i: number
  y: number
  title: string
  sub: string
  mutable?: boolean
  children?: React.ReactNode
}) {
  const x = 96 + i * 20
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={540}
        height={58}
        rx={7}
        className={mutable ? "fill-redline/6 stroke-redline" : "fill-card stroke-foreground/70"}
        strokeWidth={mutable ? 1.6 : 1.2}
      />
      <rect x={x + 10} y={y + 10} width={26} height={26} rx={5} className={mutable ? "fill-redline" : "fill-primary"} />
      <text x={x + 23} y={y + 28} textAnchor="middle" className="fill-primary-foreground font-mono text-[11px] font-semibold">
        {i + 1}
      </text>
      <text x={x + 46} y={y + 25} className="fill-foreground font-mono text-[11.5px] font-semibold">
        {title}
      </text>
      <text x={x + 46} y={y + 42} className="fill-muted-foreground font-mono text-[9.5px]">
        {sub}
      </text>
      {children}
    </g>
  )
}

export function AnatomyDiagram({ nDirectives = 10 }: { nDirectives?: number }) {
  return (
    <Fig
      title="Fig. 2 — anatomy of the prompt artifact"
      note="render order, top → bottom"
      viewBox="0 0 1160 400"
    >
      <defs>
        <marker id="anat-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M 0 1 L 9 5 L 0 9 z" className="fill-redline" />
        </marker>
      </defs>

      <Slab i={0} y={40} title="GLOBAL PREAMBLE" sub="task framing · frozen verbatim from seed" />
      <Slab i={1} y={112} title="NULL & FORMAT POLICY" sub="output contract · invariant for every candidate" />

      <Slab i={2} y={184} title="SUBTREE DIRECTIVES" sub="per-field scoped guidance — the mutation surface" mutable>
        {Array.from({ length: 5 }).map((_, k) => (
          <g key={k}>
            <rect x={430 + k * 36} y={200} width={30} height={18} rx={4} className="fill-background stroke-foreground/40" strokeWidth={1} />
            <text x={445 + k * 36} y={212.5} textAnchor="middle" className="fill-muted-foreground font-mono text-[9px]">
              ${["name", "role", "skills", "edu", "exp"][k]}
            </text>
          </g>
        ))}
        <text x={430 + 5 * 36 + 2} y={213} className="fill-muted-foreground font-mono text-[9.5px]">
          ×{nDirectives}
        </text>
      </Slab>

      <Slab i={3} y={256} title="VERIFICATION CHECKLIST" sub="self-check steps appended last" />

      <line x1={720} y1={330} x2={720} y2={40} className="stroke-foreground/40" strokeWidth={1.2} markerEnd="url(#anat-arrow)" />
      <text x={720} y={352} textAnchor="middle" className="fill-muted-foreground font-mono text-[10px]">
        one prompt → extractor · one JSON object back
      </text>

      <text x={756} y={170} className="fill-redline font-serif text-[14px] italic">
        every accepted delta lands here —
      </text>
      <text x={756} y={190} className="fill-redline font-serif text-[14px] italic">
        one $.field per iteration, rolled back by subtree
      </text>
      <path d="M 748 176 C 706 176, 706 212, 682 212" fill="none" className="stroke-redline" strokeWidth={1.5} markerEnd="url(#anat-arrow)" />

      <text x={756} y={64} className="fill-muted-foreground font-serif text-[12.5px] italic">
        frozen — mutator may not touch
      </text>
      <path d="M 748 60 C 700 60, 700 69, 642 69" fill="none" className="stroke-foreground/40" strokeWidth={1.2} strokeDasharray="4 4" />

      <text x={756} y={306} className="fill-muted-foreground font-serif text-[12.5px] italic">
        checklist grows only after a caught failure
      </text>
      <path d="M 748 302 C 700 302, 700 285, 642 285" fill="none" className="stroke-foreground/40" strokeWidth={1.2} strokeDasharray="4 4" />
    </Fig>
  )
}
