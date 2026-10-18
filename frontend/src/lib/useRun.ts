import { useEffect, useState } from "react"
import type { RunSnapshot } from "./types"

type RunState =
  | { status: "loading" }
  | { status: "ready"; run: RunSnapshot }
  | { status: "error"; error: string }

export function useRun(): RunState {
  const [state, setState] = useState<RunState>({ status: "loading" })

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const indexRes = await fetch("/data/index.json")
        if (!indexRes.ok) throw new Error(`no export found (${indexRes.status}) — run: uv run apo-extract export-web --run-id <id>`)
        const index = await indexRes.json()
        const latest = [...(index.runs ?? [])].sort(
          (a: any, b: any) => (b.exported_at ?? 0) - (a.exported_at ?? 0)
        )[0]
        if (!latest) throw new Error("index.json lists no runs — run: uv run apo-extract export-web --run-id <id>")
        const runRes = await fetch(`/data/${latest.run_id}.json`)
        if (!runRes.ok) throw new Error(`failed to load ${latest.run_id}.json (${runRes.status})`)
        const run = (await runRes.json()) as RunSnapshot
        if (!cancelled) setState({ status: "ready", run })
      } catch (err) {
        if (!cancelled) setState({ status: "error", error: err instanceof Error ? err.message : String(err) })
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  return state
}
