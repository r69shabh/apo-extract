import {
  Badge,
  badgeVariants,
} from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { fmt, type IterationRow, type RunSnapshot } from "@/lib/types"
import { cn } from "@/lib/utils"

const DECISION_STYLE: Record<string, { label: string; className: string }> = {
  ACCEPTED: { label: "accepted", className: "border-moss/50 bg-moss/10 text-moss" },
  REJECTED_STAGE1: { label: "gate 1 · train", className: "border-amberink/50 bg-amberink/10 text-amberink" },
  REJECTED_DOMINATED: { label: "dominated", className: "border-redline/40 bg-redline/5 text-redline" },
  REJECTED_NO_GAIN: { label: "no gain", className: "border-border bg-muted text-muted-foreground" },
  PENDING: { label: "pending", className: "border-border bg-muted text-muted-foreground" },
}

function DecisionBadge({ decision }: { decision: string }) {
  const meta = DECISION_STYLE[decision] ?? { label: decision.toLowerCase(), className: "" }
  return (
    <Badge variant="outline" className={cn("font-mono text-[10px]", meta.className)}>
      {meta.label}
    </Badge>
  )
}

export function IterationLedger({ run }: { run: RunSnapshot }) {
  const rows = run.iterations

  return (
    <Card className="reveal" style={{ "--i": 2 } as React.CSSProperties}>
      <CardHeader>
        <CardTitle className="font-serif text-lg">Iteration ledger</CardTitle>
        <CardDescription className="font-mono text-[11px]">
          every proposal, its evidence, and the gate that judged it
        </CardDescription>
      </CardHeader>
      <CardContent>
        {rows.length === 0 ? (
          <p className="text-muted-foreground py-6 text-center font-mono text-[11px]">
            run in progress — no iterations committed yet ({run.run.current_iteration ?? 0} in flight)
          </p>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider">iter</TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider">candidate</TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider">mutation</TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider">target</TableHead>
                <TableHead className="text-right font-mono text-[10px] uppercase tracking-wider">train F1</TableHead>
                <TableHead className="text-right font-mono text-[10px] uppercase tracking-wider">val F1</TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider">decision</TableHead>
                <TableHead className="font-mono text-[10px] uppercase tracking-wider">hypothesis</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((r) => (
                <IterationRowView key={`${r.iteration}-${r.candidate_id}`} row={r} />
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  )
}

function IterationRowView({ row }: { row: IterationRow }) {
  return (
    <TableRow className={row.decision === "ACCEPTED" ? "bg-moss/[0.04]" : undefined}>
      <TableCell className="font-mono text-[11px]">{row.iteration}</TableCell>
      <TableCell className="font-mono text-[11px] font-semibold">{row.candidate_id}</TableCell>
      <TableCell>
        <Badge variant="outline" className="font-mono text-[10px]">
          {row.mutation_type}
        </Badge>
      </TableCell>
      <TableCell className="font-mono text-[10.5px] text-muted-foreground">
        {row.target_subtree || "—"}
      </TableCell>
      <TableCell className="text-right font-mono text-[11px] tabular-nums">{fmt(row.train_f1)}</TableCell>
      <TableCell className="text-right font-mono text-[11px] tabular-nums">{fmt(row.val_f1)}</TableCell>
      <TableCell>
        <DecisionBadge decision={row.decision} />
      </TableCell>
      <TableCell className="max-w-[260px]">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="block max-w-[240px] cursor-default truncate text-[11px] text-muted-foreground">
              {row.hypothesis || row.decision_reason || "—"}
            </span>
          </TooltipTrigger>
          <TooltipContent className="max-w-sm">
            <p className="font-mono text-[11px] leading-relaxed">{row.hypothesis || "—"}</p>
            {row.decision_reason ? (
              <p className="mt-1 border-t border-primary-foreground/20 pt-1 font-mono text-[10px] opacity-80">
                {row.decision_reason}
              </p>
            ) : null}
          </TooltipContent>
        </Tooltip>
      </TableCell>
    </TableRow>
  )
}
