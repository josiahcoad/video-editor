import { useState } from "react"
import type { Hesitation } from "@/types/api"
import type { TurnEntry, TipEntry } from "@/hooks/useCoachWs"
import { cn, extractAdviceDisplay, stripCodeFences } from "@/lib/utils"

function esc(s: string): string {
  if (!s) return ""
  const div = document.createElement("div")
  div.textContent = s
  return div.innerHTML
}

function rankBadge(rank?: string): React.ReactNode {
  if (!rank || !["S", "M", "L"].includes(rank)) return null
  const titles: Record<string, string> = {
    S: "Minor, not deal-breaking",
    M: "Blocking but overcomeable",
    L: "Hard stop",
  }
  const classes: Record<string, string> = {
    S: "bg-slate-700/60 text-slate-400",
    M: "bg-amber-900/50 text-amber-300",
    L: "bg-red-900/50 text-red-400",
  }
  return (
    <span
      className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded", classes[rank])}
      title={titles[rank]}
    >
      {rank}
    </span>
  )
}

interface SessionViewProps {
  turns: TurnEntry[]
  tips: TipEntry[]
  objections: Hesitation[]
  interimText: string
  showInterim: boolean
  showScoreBar: boolean
  closeScore: number | null
  scoreBarColor: "emerald" | "amber" | "red"
  stepsToClose: number | null
  autoCoach: boolean
  critiqueMode: boolean
  coachingLoading?: boolean
  onAutoCoachChange: (v: boolean) => void
  onCritiqueChange: (v: boolean) => void
  onCoachMe: () => void
  onAskQuestion: (q: string) => void
  onCoachUpToTurn?: (turn: number) => void
  onSubmitTurn?: (text: string) => void
}

export function SessionView({
  turns,
  tips,
  objections,
  interimText,
  showInterim,
  showScoreBar,
  closeScore,
  scoreBarColor,
  stepsToClose,
  autoCoach,
  critiqueMode,
  onAutoCoachChange,
  onCritiqueChange,
  onCoachMe,
  onAskQuestion,
  onCoachUpToTurn,
  onSubmitTurn,
  coachingLoading = false,
}: SessionViewProps) {
  const [question, setQuestion] = useState("")
  const [turnText, setTurnText] = useState("")
  const scoreFillClass =
    scoreBarColor === "emerald"
      ? "bg-emerald-500"
      : scoreBarColor === "amber"
        ? "bg-amber-500"
        : "bg-red-500"

  function handleAsk() {
    const q = question.trim()
    if (q) {
      onAskQuestion(q)
      setQuestion("")
    }
  }

  function handleSubmitTurn() {
    const t = turnText.trim()
    if (t && onSubmitTurn) {
      onSubmitTurn(t)
      setTurnText("")
    }
  }

  return (
    <main className="flex-1 min-h-0 max-w-7xl w-full mx-auto p-4 grid grid-cols-1 md:grid-cols-12 grid-rows-1 gap-4">
      {/* Transcript */}
      <section className="md:col-span-5 flex flex-col min-h-0 bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
        <div className="shrink-0 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
          <h2 className="text-sm font-semibold text-slate-400">📝 Transcript</h2>
          <span className="text-xs text-slate-600">
            {turns.length} turn{turns.length !== 1 ? "s" : ""}
          </span>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-1.5">
          {turns.length === 0 ? (
            <p className="text-slate-600 text-sm italic">Transcript appears here…</p>
          ) : (
            turns.map((t) => (
              <div
                key={`${t.turn}-${t.text.slice(0, 20)}`}
                className={cn(
                  "animate-fade-up bg-slate-800/60 rounded-lg px-3 py-2",
                  t.pending && "cursor-pointer hover:bg-slate-700/70"
                )}
                onClick={() =>
                  t.pending && onCoachUpToTurn?.(t.turn)
                }
                onKeyDown={(e) =>
                  t.pending && (e.key === "Enter" || e.key === " ") && onCoachUpToTurn?.(t.turn)
                }
                role={t.pending ? "button" : undefined}
                tabIndex={t.pending ? 0 : undefined}
              >
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span className="text-[10px] font-bold text-slate-500">
                    TURN {t.turn}
                  </span>
                  <span className="text-[10px] text-slate-700">
                    {(t.confidence * 100).toFixed(0)}%
                  </span>
                  {t.pending && (
                    <span className="text-[10px] text-blue-500 ml-auto">
                      click to coach →
                    </span>
                  )}
                </div>
                <p
                  className="text-[12px] leading-relaxed text-slate-300"
                  dangerouslySetInnerHTML={{ __html: esc(t.text) }}
                />
              </div>
            ))
          )}
        </div>
        {showInterim && (
          <div className="shrink-0 px-4 py-2 border-t border-slate-800">
            <div className="flex items-center gap-2">
              <span className="flex gap-0.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-dot"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </span>
              <span className="text-sm text-slate-500 truncate">
                {interimText}
              </span>
            </div>
          </div>
        )}
        {onSubmitTurn && (
          <div className="shrink-0 px-3 py-2 border-t border-slate-800">
            <div className="flex gap-2">
              <input
                type="text"
                placeholder="Type a conversation turn…"
                value={turnText}
                onChange={(e) => setTurnText(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSubmitTurn()}
                className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-blue-600"
              />
              <button
                type="button"
                onClick={handleSubmitTurn}
                disabled={!turnText.trim()}
                className="px-3 py-1.5 text-xs rounded-lg bg-blue-600 hover:bg-blue-500 font-medium disabled:opacity-50 disabled:pointer-events-none"
              >
                Send
              </button>
            </div>
          </div>
        )}
      </section>

      {/* Coach */}
      <section className="md:col-span-4 flex flex-col min-h-0 bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
        <div className="shrink-0 px-4 py-2 border-b border-slate-800 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h2 className="text-sm font-semibold text-emerald-400">🎯 Coach</h2>
            <label className="flex items-center gap-1.5 cursor-pointer">
              <input
                type="checkbox"
                checked={autoCoach}
                onChange={(e) => onAutoCoachChange(e.target.checked)}
                className="w-3.5 h-3.5 rounded accent-emerald-500"
              />
              <span className="text-[11px] text-slate-500">Auto</span>
            </label>
            <label
              className="flex items-center gap-1.5 cursor-pointer"
              title="Actor suggests several next steps; critic scores each for P(close). Slower."
            >
              <input
                type="checkbox"
                checked={critiqueMode}
                onChange={(e) => onCritiqueChange(e.target.checked)}
                className="w-3.5 h-3.5 rounded accent-purple-500"
              />
              <span className="text-[11px] text-slate-500">Critique</span>
            </label>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-600">
              {tips.length} tip{tips.length !== 1 ? "s" : ""}
            </span>
            {!autoCoach && (
              <button
                type="button"
                onClick={() => onCoachMe()}
                className="px-2.5 py-1 text-[11px] rounded-md bg-emerald-700 hover:bg-emerald-600 font-medium"
              >
                Coach Me
              </button>
            )}
          </div>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-2">
          {coachingLoading && (
            <div className="flex items-center gap-2 text-slate-500 text-sm py-2">
              <span className="flex gap-0.5">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-dot" />
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-dot" style={{ animationDelay: "0.15s" }} />
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse-dot" style={{ animationDelay: "0.3s" }} />
              </span>
              Thinking…
            </div>
          )}
          {tips.length === 0 && !coachingLoading ? (
            <p className="text-slate-600 text-sm italic">Coaching appears here…</p>
          ) : (
            tips.map((tip) => (
              <TipCard key={tip.id} tip={tip} esc={esc} />
            ))
          )}
        </div>
        {showScoreBar && (
          <div className="shrink-0 px-4 py-2 border-t border-slate-800">
            <div className="flex items-center justify-between text-[11px] mb-1">
              <span className="text-slate-500">Close probability</span>
              <span className="font-bold text-emerald-400">
                {closeScore != null ? `${closeScore}%` : "—"}
              </span>
            </div>
            <div className="w-full bg-slate-800 rounded-full h-1.5">
              <div
                className={cn(
                  scoreFillClass,
                  "h-1.5 rounded-full transition-all duration-500"
                )}
                style={{ width: `${closeScore ?? 0}%` }}
              />
            </div>
            <div className="flex items-center justify-between text-[11px] mt-1">
              <span className="text-slate-500">Steps to close</span>
              <span className="font-bold text-amber-400">
                {stepsToClose != null ? stepsToClose : "—"}
              </span>
            </div>
          </div>
        )}
        <div className="shrink-0 px-3 py-2 border-t border-slate-800">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Ask your coach a question…"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleAsk()}
              className="flex-1 bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:border-emerald-600"
            />
            <button
              type="button"
              onClick={handleAsk}
              className="px-3 py-1.5 text-xs rounded-lg bg-slate-700 hover:bg-slate-600 font-medium"
            >
              Ask
            </button>
          </div>
        </div>
      </section>

      {/* Hesitations */}
      <section className="md:col-span-3 flex flex-col min-h-0 bg-slate-900 rounded-xl border border-slate-800 overflow-hidden">
        <div className="shrink-0 px-4 py-2 border-b border-slate-800">
          <h2 className="text-sm font-semibold text-amber-400">⚠️ Hesitations</h2>
        </div>
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-1.5">
          {objections.length === 0 ? (
            <p className="text-slate-600 text-sm italic">Hesitations tracked here…</p>
          ) : (
            objections.map((o, i) => {
              const resolved = o.status === "resolved"
              const suggestion =
                o.resolution_suggestion?.trim()
              return (
                <div
                  key={`${o.text}-${i}`}
                  className={cn(
                    "rounded-lg px-3 py-2 border text-[12px]",
                    resolved
                      ? "border-emerald-900/30 bg-emerald-950/20 text-emerald-400 line-through opacity-60"
                      : "border-amber-900/40 bg-amber-950/20 text-amber-300"
                  )}
                >
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span>{resolved ? "✅" : "⚠️"}</span>
                    {rankBadge(o.rank)}
                    <span dangerouslySetInnerHTML={{ __html: esc(o.text) }} />
                  </div>
                  {suggestion && (
                    <div className="mt-1 pl-5 text-[11px] text-slate-400">
                      ↪ <span dangerouslySetInnerHTML={{ __html: esc(suggestion) }} />
                    </div>
                  )}
                </div>
              )
            })
          )}
        </div>
      </section>
    </main>
  )
}

function TipCard({
  tip,
  esc,
}: {
  tip: TipEntry
  esc: (s: string) => string
}) {
  const isError = tip.advice.startsWith("Error") || tip.number === 0
  if (isError) {
    return (
      <div className="animate-fade-up rounded-lg px-3 py-2 bg-red-950/40 border border-red-900/40">
        <p className="text-[12px] text-red-400" dangerouslySetInnerHTML={{ __html: esc(extractAdviceDisplay(tip.advice)) }} />
      </div>
    )
  }
  if (tip.candidates && tip.candidates.length > 0) {
    return (
      <div className="animate-fade-up rounded-lg px-3 py-2.5 border border-purple-900/40 bg-purple-950/20">
        <div className="flex items-baseline gap-2 mb-2">
          <span className="text-[10px] font-bold text-purple-400">
            🧠 Critique #{tip.number}
          </span>
          <span className="text-[10px] text-slate-700">turn {tip.turn}</span>
          {tip.closeScore != null && tip.closeScore >= 0 && (
            <span className="text-[10px] text-slate-600 ml-auto">
              Best: {tip.closeScore}%
            </span>
          )}
        </div>
        {tip.candidates.map((c, i) => {
          const isTop = i === 0
          const prob =
            c.close_probability != null && c.close_probability >= 0
              ? `${c.close_probability}%`
              : "—"
          return (
            <div
              key={i}
              className={i > 0 ? "mt-2 pt-2 border-t border-slate-700/50" : ""}
            >
              <div className="flex items-baseline gap-2 mb-0.5">
                <span
                  className={cn(
                    "text-[10px] font-bold",
                    isTop ? "text-purple-400" : "text-slate-500"
                  )}
                >
                  #{i + 1} {c.one_liner ? esc(stripCodeFences(c.one_liner)) : ""}
                </span>
                <span className="text-[10px] text-emerald-400 ml-auto">
                  P(close) {prob}
                </span>
              </div>
              <p
                className={cn(
                  "text-[12px] leading-relaxed",
                  isTop ? "text-purple-200 font-medium" : "text-slate-400"
                )}
                dangerouslySetInnerHTML={{ __html: esc(stripCodeFences(c.action)) }}
              />
              {c.trajectory_notes && (
                <p
                  className="text-[11px] text-slate-500 mt-0.5 italic"
                  dangerouslySetInnerHTML={{ __html: esc(stripCodeFences(c.trajectory_notes)) }}
                />
              )}
            </div>
          )
        })}
      </div>
    )
  }
  const border = tip.isCustom
    ? "border-blue-900/40 bg-blue-950/30"
    : "border-emerald-900/40 bg-emerald-950/30"
  const label = tip.isCustom ? "💬" : "🎯"
  const labelColor = tip.isCustom ? "text-blue-400" : "text-emerald-500"
  return (
    <div className={cn("animate-fade-up rounded-lg px-3 py-2.5 border", border)}>
      <div className="flex items-baseline gap-2 mb-1">
        <span className={cn("text-[10px] font-bold", labelColor)}>
          {label} #{tip.number}
        </span>
        <span className="text-[10px] text-slate-700">turn {tip.turn}</span>
        {(tip.closeScore != null && tip.closeScore >= 0) ||
        (tip.stepsToClose != null && tip.stepsToClose >= 0) ? (
          <span className="text-[10px] text-slate-600 ml-auto">
            {tip.closeScore != null ? `${tip.closeScore}%` : ""}
            {tip.closeScore != null && tip.stepsToClose != null ? " · " : ""}
            {tip.stepsToClose != null ? `${tip.stepsToClose} steps` : ""}
          </span>
        ) : null}
      </div>
      <p
        className="text-[12px] leading-relaxed text-emerald-300 font-medium"
        dangerouslySetInnerHTML={{ __html: esc(extractAdviceDisplay(tip.advice)) }}
      />
    </div>
  )
}
