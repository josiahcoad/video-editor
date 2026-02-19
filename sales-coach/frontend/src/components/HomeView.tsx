import { useCallback, useEffect, useMemo, useState } from "react"
import {
  Bar,
  BarChart,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import type { HomeData, PerformanceReview, CallsPerDay, TodoItem, TodoType } from "@/types/api"
import { fetchHome, fetchTodos, createTodo, deleteTodo, suggestTodos, generatePerformanceReview } from "@/lib/api"
import { cn } from "@/lib/utils"
import { RefreshCw, Trash2, Sparkles } from "lucide-react"

const STATUS_LABELS: Record<string, string> = {
  prospect: "Prospect",
  interested: "Interested",
  demoing: "Demoing",
  reviewing_contract: "Reviewing Contract",
  closed_won: "Closed Won",
  closed_lost: "Closed Lost",
}

const STATUS_COLORS: Record<string, string> = {
  prospect: "bg-slate-600",
  interested: "bg-blue-600",
  demoing: "bg-purple-600",
  reviewing_contract: "bg-amber-600",
  closed_won: "bg-emerald-600",
  closed_lost: "bg-red-600",
}

const STATUS_ORDER = [
  "prospect",
  "interested",
  "demoing",
  "reviewing_contract",
  "closed_won",
  "closed_lost",
]

const CHART_DAYS = 14

/** Group conversation start times (ISO) by browser local date. */
function buildCallsPerDayLocal(startTimes: string[], days: number = 30): CallsPerDay[] {
  const byDay: Record<string, number> = {}
  for (const iso of startTimes) {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) continue
    const y = d.getFullYear()
    const m = String(d.getMonth() + 1).padStart(2, "0")
    const day = String(d.getDate()).padStart(2, "0")
    const key = `${y}-${m}-${day}`
    byDay[key] = (byDay[key] ?? 0) + 1
  }
  const today = new Date()
  const result: CallsPerDay[] = []
  for (let i = days - 1; i >= 0; i--) {
    const d = new Date(today)
    d.setDate(d.getDate() - i)
    const y = d.getFullYear()
    const m = String(d.getMonth() + 1).padStart(2, "0")
    const day = String(d.getDate()).padStart(2, "0")
    const dateStr = `${y}-${m}-${day}`
    result.push({ date: dateStr, count: byDay[dateStr] ?? 0 })
  }
  return result
}

function formatChartDate(dateStr: string): string {
  const [y, m, d] = dateStr.split("-").map(Number)
  const date = new Date(y, m - 1, d)
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
}

const CHART_HEIGHT = 200
const SLATE_600 = "#475569"
const CYAN_500 = "#06b6d4"
const SLATE_800 = "#1e293b"

function CallsPerDayChart({ callsPerDay }: { callsPerDay: CallsPerDay[] }) {
  const slice = callsPerDay.slice(-CHART_DAYS)
  const today = new Date()
  const todayStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`

  const chartData = slice.map((d) => ({
    ...d,
    label: formatChartDate(d.date),
    isToday: d.date === todayStr,
  }))

  return (
    <div className="space-y-1">
      <p className="text-xs text-slate-500">Calls per day (last {CHART_DAYS} days)</p>
      <div style={{ width: "100%", height: CHART_HEIGHT }}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="label"
              tick={{ fill: "#64748b", fontSize: 10 }}
              axisLine={{ stroke: "#334155" }}
              tickLine={{ stroke: "#334155" }}
            />
            <YAxis
              allowDecimals={false}
              tick={{ fill: "#64748b", fontSize: 10 }}
              axisLine={{ stroke: "#334155" }}
              tickLine={{ stroke: "#334155" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: SLATE_800,
                border: "1px solid #334155",
                borderRadius: "8px",
              }}
              labelStyle={{ color: "#94a3b8" }}
              formatter={(value: number) => [`${value} call${value !== 1 ? "s" : ""}`, "Calls"]}
              labelFormatter={(label) => label}
            />
            <Bar dataKey="count" radius={[4, 4, 0, 0]} maxBarSize={40}>
              {chartData.map((entry, index) => (
                <Cell
                  key={entry.date}
                  fill={entry.count > 0 ? (entry.isToday ? CYAN_500 : SLATE_600) : `${SLATE_800}80`}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function PipelineBar({
  pipeline,
  totalContacts,
}: {
  pipeline: HomeData["pipeline"]
  totalContacts: number
}) {
  const byStatus = Object.fromEntries(pipeline.map((s) => [s.status, s.count]))
  return (
    <div className="space-y-2">
      {STATUS_ORDER.map((status) => {
        const count = byStatus[status] || 0
        const pct = totalContacts > 0 ? (count / totalContacts) * 100 : 0
        return (
          <div key={status} className="flex items-center gap-3">
            <span className="w-36 text-xs text-slate-400 text-right shrink-0">
              {STATUS_LABELS[status] || status}
            </span>
            <div className="flex-1 bg-slate-800 rounded-full h-6 overflow-hidden relative">
              <div
                className={cn(
                  "h-full rounded-full transition-all duration-500",
                  STATUS_COLORS[status] || "bg-slate-600"
                )}
                style={{ width: `${Math.max(pct, count > 0 ? 4 : 0)}%` }}
              />
              {count > 0 && (
                <span className="absolute inset-y-0 left-2 flex items-center text-[11px] font-semibold text-white">
                  {count}
                </span>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ReviewSection({
  review,
  generatedAt,
  onGenerate,
  generating,
}: {
  review: PerformanceReview | null
  generatedAt: string | null
  onGenerate: () => void
  generating: boolean
}) {
  return (
    <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-slate-200">
          Performance Review
        </h3>
        <button
          type="button"
          onClick={onGenerate}
          disabled={generating}
          className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-medium rounded-lg bg-slate-800 text-slate-300 hover:bg-slate-700 border border-slate-700 disabled:opacity-50"
        >
          <RefreshCw className={cn("w-3.5 h-3.5", generating && "animate-spin")} />
          {generating
            ? "Generating…"
            : review
              ? "Regenerate"
              : "Generate Review"}
        </button>
      </div>

      {!review && !generating && (
        <p className="text-sm text-slate-500 italic">
          No review yet. Click "Generate Review" to have AI analyze all your
          conversations and provide feedback.
        </p>
      )}

      {review && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <span className="text-3xl font-bold text-white">
              {review.score}
              <span className="text-lg text-slate-500">/10</span>
            </span>
            {generatedAt && (
              <span className="text-[10px] text-slate-600">
                Generated{" "}
                {new Date(generatedAt).toLocaleDateString("en-US", {
                  month: "short",
                  day: "numeric",
                  year: "2-digit",
                })}
              </span>
            )}
          </div>

          <p className="text-sm text-slate-300 leading-relaxed">
            {review.overall_assessment}
          </p>

          <ReviewList
            title="Strengths"
            items={review.strengths}
            color="text-emerald-400"
            icon="+"
          />
          <ReviewList
            title="Areas for Improvement"
            items={review.areas_for_improvement}
            color="text-amber-400"
            icon="→"
          />
          <ReviewList
            title="Common Mistakes"
            items={review.common_mistakes}
            color="text-red-400"
            icon="✕"
          />
          <ReviewList
            title="Coaching Recommendations"
            items={review.coaching_recommendations}
            color="text-cyan-400"
            icon="💡"
          />
        </div>
      )}
    </div>
  )
}

function ReviewList({
  title,
  items,
  color,
  icon,
}: {
  title: string
  items: string[]
  color: string
  icon: string
}) {
  if (!items || items.length === 0) return null
  return (
    <div>
      <h4 className={cn("text-xs font-semibold uppercase tracking-wide mb-1.5", color)}>
        {title}
      </h4>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="text-sm text-slate-400 flex gap-2">
            <span className={cn("shrink-0", color)}>{icon}</span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export function HomeView() {
  const [data, setData] = useState<HomeData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [generating, setGenerating] = useState(false)
  const [todos, setTodos] = useState<TodoItem[] | null>(null)
  const [todosLoading, setTodosLoading] = useState(false)
  const [newTodoType, setNewTodoType] = useState<TodoType>("call")
  const [newTodoTitle, setNewTodoTitle] = useState("")
  const [addingTodo, setAddingTodo] = useState(false)
  const [suggesting, setSuggesting] = useState(false)

  const load = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      setData(await fetchHome())
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load")
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    load()
  }, [load])

  const loadTodos = useCallback(async () => {
    if (!data?.seller?.id) return
    setTodosLoading(true)
    try {
      const res = await fetchTodos(data.seller.id)
      setTodos(res.todos ?? [])
    } catch {
      setTodos([])
    } finally {
      setTodosLoading(false)
    }
  }, [data?.seller?.id])

  useEffect(() => {
    if (data?.seller?.id && todos === null) loadTodos()
  }, [data?.seller?.id, todos, loadTodos])

  const handleAddTodo = async () => {
    if (!data?.seller?.id || !newTodoTitle.trim() || addingTodo) return
    setAddingTodo(true)
    try {
      await createTodo(data.seller.id, {
        type: newTodoType,
        title: newTodoTitle.trim(),
      })
      setNewTodoTitle("")
      setTodos(null)
      await loadTodos()
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to add todo")
    } finally {
      setAddingTodo(false)
    }
  }

  const handleDeleteTodo = async (id: number) => {
    try {
      await deleteTodo(id)
      setTodos((prev) => (prev ?? []).filter((t) => t.id !== id))
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to delete")
    }
  }

  const handleSuggestTodos = async () => {
    if (!data?.seller?.id || suggesting) return
    setSuggesting(true)
    try {
      const res = await suggestTodos(data.seller.id)
      setTodos(res.todos)
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to suggest tasks")
    } finally {
      setSuggesting(false)
    }
  }

  const callsPerDay = useMemo(
    () => buildCallsPerDayLocal(data?.conversation_start_times ?? [], 30),
    [data?.conversation_start_times]
  )

  const handleGenerate = async () => {
    if (!data) return
    setGenerating(true)
    try {
      const review = await generatePerformanceReview(data.seller.id)
      setData((prev) =>
        prev
          ? {
              ...prev,
              performance_review: review,
              review_generated_at: new Date().toISOString(),
            }
          : prev
      )
    } catch (e) {
      alert(e instanceof Error ? e.message : "Failed to generate review")
    } finally {
      setGenerating(false)
    }
  }

  if (loading) {
    return (
      <main className="flex-1 min-h-0 max-w-5xl w-full mx-auto p-4 overflow-y-auto">
        <p className="text-slate-600 italic text-sm">Loading…</p>
      </main>
    )
  }
  if (error || !data) {
    return (
      <main className="flex-1 min-h-0 max-w-5xl w-full mx-auto p-4 overflow-y-auto">
        <p className="text-red-400 text-sm">{error || "No data"}</p>
      </main>
    )
  }

  return (
    <main className="flex-1 min-h-0 max-w-5xl w-full mx-auto p-4 overflow-y-auto space-y-6">
      <div>
        <h2 className="text-xl font-bold text-white">
          Welcome back, {data.seller.first_name}
        </h2>
        <p className="text-sm text-slate-500 mt-1">
          {data.total_contacts} contacts · {data.total_conversations} conversations
        </p>
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-slate-200">
            Quick actions
          </h3>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleSuggestTodos}
              disabled={suggesting}
              className="inline-flex items-center gap-1.5 text-sm text-amber-400 hover:text-amber-300 disabled:opacity-50"
              aria-label="AI suggest tasks"
            >
              <Sparkles className={cn("size-4", suggesting && "animate-pulse")} />
              {suggesting ? "Thinking…" : "Suggest"}
            </button>
            <button
              type="button"
              onClick={() => { setTodos(null); loadTodos() }}
              disabled={todosLoading}
              className="inline-flex items-center gap-1.5 text-sm text-slate-400 hover:text-slate-200 disabled:opacity-50"
              aria-label="Refresh todos"
            >
              <RefreshCw className={cn("size-4", todosLoading && "animate-spin")} />
            </button>
          </div>
        </div>
        {todosLoading && todos === null ? (
          <p className="text-sm text-slate-500 italic">Loading…</p>
        ) : (
          <>
            <ul className="list-none space-y-2 text-sm text-slate-300">
              {(todos ?? []).map((todo) => (
                <li
                  key={todo.id}
                  className="flex items-center gap-2 group rounded-lg bg-slate-800/60 px-3 py-2"
                >
                  <span
                    className={cn(
                      "shrink-0 text-[10px] font-semibold uppercase px-1.5 py-0.5 rounded",
                      todo.type === "email" && "bg-blue-900/50 text-blue-300",
                      todo.type === "call" && "bg-emerald-900/50 text-emerald-300",
                      todo.type === "other" && "bg-slate-600 text-slate-300"
                    )}
                  >
                    {todo.type}
                  </span>
                  <span className="flex-1 min-w-0">
                    {todo.contact_name && (
                      <span className="text-slate-500 text-xs mr-1.5">{todo.contact_name} ·</span>
                    )}
                    {todo.title}
                  </span>
                  <button
                    type="button"
                    onClick={() => handleDeleteTodo(todo.id)}
                    className="shrink-0 p-1 rounded text-slate-500 hover:text-red-400 hover:bg-slate-700 opacity-0 group-hover:opacity-100 transition"
                    aria-label="Delete todo"
                  >
                    <Trash2 className="size-3.5" />
                  </button>
                </li>
              ))}
            </ul>
            <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-800">
              <select
                value={newTodoType}
                onChange={(e) => setNewTodoType(e.target.value as TodoType)}
                className="bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-sm text-slate-200 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                <option value="email">Email</option>
                <option value="call">Call</option>
                <option value="other">Other</option>
              </select>
              <input
                type="text"
                placeholder="Add a task…"
                value={newTodoTitle}
                onChange={(e) => setNewTodoTitle(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleAddTodo()}
                className="flex-1 min-w-[140px] bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
              <button
                type="button"
                onClick={handleAddTodo}
                disabled={!newTodoTitle.trim() || addingTodo}
                className="px-3 py-1.5 text-sm font-medium rounded-lg bg-cyan-700 text-white hover:bg-cyan-600 disabled:opacity-50 disabled:pointer-events-none"
              >
                {addingTodo ? "Adding…" : "Add"}
              </button>
            </div>
            {(todos ?? []).length === 0 && (
              <p className="text-sm text-slate-500 italic">
                Add tasks above (email, call, or other).
              </p>
            )}
          </>
        )}
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 className="text-base font-semibold text-slate-200">
          Calls per day
        </h3>
        <CallsPerDayChart callsPerDay={callsPerDay} />
      </div>

      <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 space-y-3">
        <h3 className="text-base font-semibold text-slate-200">
          Pipeline Breakdown
        </h3>
        {data.total_contacts === 0 ? (
          <p className="text-sm text-slate-500 italic">
            No contacts yet. Add some from the Contacts tab.
          </p>
        ) : (
          <PipelineBar
            pipeline={data.pipeline}
            totalContacts={data.total_contacts}
          />
        )}
      </div>

      <ReviewSection
        review={data.performance_review}
        generatedAt={data.review_generated_at}
        onGenerate={handleGenerate}
        generating={generating}
      />
    </main>
  )
}
