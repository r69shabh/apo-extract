import type { ReactNode } from "react"

export function Fig({
  title,
  note,
  viewBox,
  children,
  className,
}: {
  title: string
  note?: string
  viewBox: string
  children: ReactNode
  className?: string
}) {
  return (
    <figure className={className}>
      <div className="mb-3 flex items-baseline justify-between gap-4">
        <figcaption className="kicker">{title}</figcaption>
        {note ? (
          <span className="font-mono text-[10px] text-muted-foreground">{note}</span>
        ) : null}
      </div>
      <div className="overflow-hidden rounded-lg border bg-[oklch(0.992_0.004_88)] px-2 py-3">
        <svg viewBox={viewBox} className="h-auto w-full" role="img" aria-label={title}>
          {children}
        </svg>
      </div>
    </figure>
  )
}
