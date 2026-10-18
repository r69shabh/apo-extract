import { useState } from "react"
import { BookOpen, FileText, PenTool } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TooltipProvider } from "@/components/ui/tooltip"
import { timeAgo, type RunSnapshot } from "@/lib/types"
import { useRun } from "@/lib/useRun"
import { DocsView } from "@/views/DocsView"
import { RunView } from "@/views/RunView"

function StatusBadge({ status }: { status: string }) {
  if (status === "running") {
    return (
      <span className="inline-flex items-center gap-1.5 font-mono text-[11px] font-semibold text-amberink">
        <span className="pulse-dot size-2 rounded-full bg-amberink" />
        running
      </span>
    )
  }
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-[11px] font-semibold text-moss">
      <span className="size-2 rounded-full bg-moss" />
      {status}
    </span>
  )
}

function TitleBlock({ run }: { run: RunSnapshot }) {
  const llms = run.config?.llms ?? {}
  const split = run.dataset.split
  const cells: Array<[string, React.ReactNode]> = [
    ["run", <span key="v" className="font-semibold">{run.run_id}</span>],
    ["status", <StatusBadge key="v" status={run.run.status} />],
    ["dataset", run.dataset.name ?? run.dataset.path ?? "—"],
    ["schema", `${run.dataset.schema_fields.length} fields · ${run.dataset.n_documents} docs`],
    ["extractor", (llms.extractor?.model ?? "—").replace(/^gemini\//, "")],
    ["judge", (run.config?.scoring?.judge_model ?? "—").replace(/^gemini\//, "")],
    ["split", `tr ${split.train.length} · va ${split.val.length} · te ${split.test.length}`],
    ["exported", `${timeAgo(run.exported_at)}`],
  ]
  return (
    <div className="title-block reveal">
      {cells.map(([k, v]) => (
        <div key={k} className="title-cell">
          <div className="kicker text-[9px]">{k}</div>
          <div className="mt-0.5 truncate font-mono text-[11px]">{v}</div>
        </div>
      ))}
    </div>
  )
}

function LoadingPage() {
  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-64 rounded-xl" />
        <Skeleton className="h-64 rounded-xl" />
      </div>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        {[0, 1, 2, 3].map((i) => (
          <Skeleton key={i} className="h-28 rounded-xl" />
        ))}
      </div>
      <Skeleton className="h-72 rounded-xl" />
    </div>
  )
}

function ErrorPage({ error }: { error: string }) {
  return (
    <Card className="mx-auto max-w-xl py-10">
      <CardContent className="space-y-4 text-center">
        <PenTool className="mx-auto size-6 text-redline" />
        <h2 className="font-serif text-xl font-semibold">No run snapshot to draw</h2>
        <p className="text-muted-foreground font-mono text-[12px]">{error}</p>
        <pre className="bg-muted rounded-md p-3 text-left font-mono text-[11.5px]">
          uv run apo-extract export-web --run-id &lt;run-id&gt;
        </pre>
        <p className="text-muted-foreground text-[12px]">
          The export writes <span className="font-mono">frontend/public/data/</span>; reload this page afterwards.
        </p>
      </CardContent>
    </Card>
  )
}

export default function App() {
  const state = useRun()
  const [tab, setTab] = useState("run")
  const run = state.status === "ready" ? state.run : null

  return (
    <TooltipProvider>
      <div className="min-h-screen">
        <header className="bg-background/85 sticky top-0 z-40 border-b backdrop-blur">
          <div className="mx-auto flex max-w-[1240px] flex-wrap items-center justify-between gap-4 px-6 py-4">
            <div className="flex items-center gap-3">
              <svg viewBox="0 0 32 32" className="size-9 rounded-md border bg-card p-1" aria-hidden>
                <path d="M8 24 L24 8" className="stroke-redline" strokeWidth={3} strokeLinecap="round" />
                <path d="M8 8 h10 M8 14 h16 M8 20 h16" className="stroke-foreground" strokeWidth={2.4} strokeLinecap="round" opacity="0.75" />
              </svg>
              <div>
                <div className="font-serif text-[22px] leading-none font-black tracking-tight">
                  APO<span className="text-redline">·</span>EXTRACT
                </div>
                <div className="kicker mt-1 text-[9.5px]">
                  automated prompt optimization — extractbench workbench
                </div>
              </div>
            </div>
            <div className="flex items-center gap-4">
              {run ? (
                <Badge variant="outline" className="hidden font-mono text-[10px] sm:inline-flex">
                  exported {timeAgo(run.exported_at)}
                </Badge>
              ) : null}
              <Tabs value={tab} onValueChange={setTab}>
                <TabsList>
                  <TabsTrigger value="run" className="gap-1.5">
                    <FileText className="size-3.5" />
                    run ledger
                  </TabsTrigger>
                  <TabsTrigger value="docs" className="gap-1.5">
                    <BookOpen className="size-3.5" />
                    how it works
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
          </div>
          {run ? (
            <div className="mx-auto max-w-[1240px] px-6 pb-3">
              <TitleBlock run={run} />
            </div>
          ) : null}
        </header>

        <main className="mx-auto max-w-[1240px] px-6 py-8">
          {state.status === "loading" ? <LoadingPage /> : null}
          {state.status === "error" ? <ErrorPage error={state.error} /> : null}
          {state.status === "ready" ? (
            <>
              <Tabs value={tab} onValueChange={setTab}>
                <TabsContent value="run" className="mt-0">
                  <RunView run={state.run} />
                </TabsContent>
                <TabsContent value="docs" className="mt-0">
                  <DocsView run={state.run} />
                </TabsContent>
              </Tabs>
            </>
          ) : null}
        </main>

        <footer className="mx-auto max-w-[1240px] px-6 pb-10">
          <Separator className="mb-4" />
          <div className="text-muted-foreground flex flex-wrap items-center justify-between gap-2 font-mono text-[10.5px]">
            <span>
              static snapshot — regenerate with <span className="text-foreground">apo-extract export-web</span> after
              the run commits new iterations
            </span>
            <span>kill -9 friendly · resume replays caches · test split scored once</span>
          </div>
        </footer>
      </div>
    </TooltipProvider>
  )
}
