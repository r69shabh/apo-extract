import { IterationLedger } from "@/components/run/IterationLedger"
import { PromptIO } from "@/components/run/PromptIO"
import { RunConfig } from "@/components/run/RunConfig"
import { ScoreCharts } from "@/components/run/ScoreCharts"
import { StatsRow } from "@/components/run/StatsRow"
import { SubtreeBoard } from "@/components/run/SubtreeBoard"
import type { RunSnapshot } from "@/lib/types"

export function RunView({ run }: { run: RunSnapshot }) {
  return (
    <div className="space-y-6">
      <PromptIO run={run} />
      <StatsRow run={run} />
      <IterationLedger run={run} />
      <ScoreCharts run={run} />
      <SubtreeBoard run={run} />
      <RunConfig run={run} />
    </div>
  )
}
