import type { ConversationDetail as ConvDetailType, Review } from "@/types/api"
import { postConversationReview } from "@/lib/api"
import { cn, extractAdviceDisplay } from "@/lib/utils"
import { type ReactNode, useState } from "react"

function esc(s: string): string {
  if (!s) return ""
  const div = document.createElement("div")
  div.textContent = s
  return div.innerHTML
}

/** Escape and show newlines; break between "), (" so numbered lists (1) (2) display on separate lines. */
function formatParagraphWithBreaks(s: string): string {
  if (!s) return ""
  let out = esc(s).replace(/\n/g, "<br />")
  out = out.replace(/\), \(/g, "),<br /><br />(")
  return out
}

function RankBadge({ rank }: { rank?: string }) {
  if (!rank || !["S", "M", "L"].includes(rank)) return null
  const classes: Record<string, string> = {
    S: "bg-slate-700/60 text-slate-400",
    M: "bg-amber-900/50 text-amber-300",
    L: "bg-red-900/50 text-red-400",
  }
  return (
    <span
      className={cn("text-[10px] font-bold px-1.5 py-0.5 rounded", classes[rank])}
    >
      {rank}
    </span>
  )
}

/** Reusable section card wrapper for review blocks */
function ReviewSectionCard({
  title,
  titleClassName,
  children,
  className,
}: {
  title: string
  titleClassName?: string
  children: ReactNode
  className?: string
}) {
  return (
    <div
      className={cn(
        "rounded-xl border border-slate-700/60 bg-slate-800/50 p-4",
        className
      )}
    >
      <p className={cn("text-[10px] font-bold uppercase tracking-wide mb-3", titleClassName)}>
        {title}
      </p>
      {children}
    </div>
  )
}

/** Single bullet as a small card to break up lists */
function ReviewBulletCard({
  children,
  borderClassName,
  className,
}: {
  children: ReactNode
  borderClassName?: string
  className?: string
}) {
  const isHtml = typeof children === "string"
  return (
    <div
      className={cn(
        "rounded-lg border-l-2 bg-slate-800/40 py-2 px-3 text-[12px] text-slate-300",
        borderClassName,
        className
      )}
      {...(isHtml ? { dangerouslySetInnerHTML: { __html: children } } : {})}
    >
      {!isHtml ? children : null}
    </div>
  )
}

function ReviewBlock({ review }: { review: Review }) {
  const score =
    typeof review.score === "number" && Number.isFinite(review.score)
      ? review.score
      : null
  const scoreColor =
    score != null
      ? score >= 7
        ? "text-emerald-400"
        : score >= 4
          ? "text-amber-400"
          : "text-red-400"
      : "text-slate-500"

  const wentWell = review.went_well || []
  const improve = review.improve || []
  const uncovered = review.hesitations_summary?.uncovered || []
  const handled = review.hesitations_summary?.handled || []
  const remaining = review.hesitations_summary?.remaining || []
  const followUps = review.follow_ups || []

  return (
    <div className="space-y-4">
      {/* Conversation performance score card */}
      <div className="rounded-xl border border-slate-700/60 bg-slate-800/50 p-4">
        <p className="text-[10px] font-bold uppercase tracking-wide text-slate-500 mb-1">
          Conversation performance
        </p>
        <span className={cn("text-2xl font-black", scoreColor)}>
          {score != null ? `${score}/10` : "—/10"}
        </span>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ReviewSectionCard title="Went well" titleClassName="text-emerald-500">
          <div className="space-y-2">
            {wentWell.length === 0 && <p className="text-[12px] text-slate-600">—</p>}
            {wentWell.map((w, i) => (
              <ReviewBulletCard key={i} borderClassName="border-l-emerald-700">
                {esc(w)}
              </ReviewBulletCard>
            ))}
          </div>
        </ReviewSectionCard>
        <ReviewSectionCard title="Improve" titleClassName="text-amber-500">
          <div className="space-y-2">
            {improve.length === 0 && <p className="text-[12px] text-slate-600">—</p>}
            {improve.map((w, i) => (
              <ReviewBulletCard key={i} borderClassName="border-l-amber-700">
                {esc(w)}
              </ReviewBulletCard>
            ))}
          </div>
        </ReviewSectionCard>
      </div>

      {/* Constraints: uncovered / handled / remaining */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ReviewSectionCard title="Uncovered" titleClassName="text-slate-400">
          <div className="space-y-2">
            {uncovered.length === 0 && <p className="text-[11px] text-slate-600">—</p>}
            {uncovered.map((h, i) => (
              <div
                key={i}
                className="rounded-lg bg-slate-800/40 py-2 px-3 text-[11px] text-slate-300"
                dangerouslySetInnerHTML={{ __html: esc(h) }}
              />
            ))}
          </div>
        </ReviewSectionCard>
        <ReviewSectionCard title="Handled" titleClassName="text-emerald-600">
          <div className="space-y-2">
            {handled.length === 0 && <p className="text-[11px] text-slate-600">—</p>}
            {handled.map((h, i) => (
              <div
                key={i}
                className="rounded-lg bg-emerald-950/30 border border-emerald-900/40 py-2 px-3 text-[11px] text-emerald-300"
                dangerouslySetInnerHTML={{ __html: esc(h) }}
              />
            ))}
          </div>
        </ReviewSectionCard>
        <ReviewSectionCard title="Remaining" titleClassName="text-red-500">
          <div className="space-y-2">
            {remaining.length === 0 && <p className="text-[11px] text-slate-600">—</p>}
            {remaining.map((h, i) => (
              <div
                key={i}
                className="rounded-lg bg-red-950/20 border border-red-900/40 py-2 px-3 text-[11px] text-red-300"
                dangerouslySetInnerHTML={{ __html: esc(h) }}
              />
            ))}
          </div>
        </ReviewSectionCard>
      </div>

      {review.key_moment && (
        <ReviewSectionCard title="Key moment" titleClassName="text-blue-400">
          <p
            className="text-[12px] text-slate-300 rounded-lg bg-blue-950/20 border border-blue-900/40 p-3"
            dangerouslySetInnerHTML={{ __html: esc(review.key_moment) }}
          />
        </ReviewSectionCard>
      )}

      <ReviewSectionCard title="Follow-ups" titleClassName="text-cyan-400">
        <div className="space-y-2">
          {followUps.length === 0 && <p className="text-[12px] text-slate-600">—</p>}
          {followUps.map((f, i) => (
            <div
              key={i}
              className="flex gap-2 rounded-lg bg-slate-800/40 py-2 px-3 text-[12px] text-slate-300"
            >
              <span className="text-cyan-500 font-semibold shrink-0">{i + 1}.</span>
              <span dangerouslySetInnerHTML={{ __html: esc(f) }} />
            </div>
          ))}
        </div>
      </ReviewSectionCard>

      {review.next_call_prep && (
        <ReviewSectionCard title="Next call prep" titleClassName="text-slate-400">
          <p
            className="text-[12px] text-slate-300 rounded-lg bg-slate-800/40 p-3"
            dangerouslySetInnerHTML={{
              __html: formatParagraphWithBreaks(review.next_call_prep),
            }}
          />
        </ReviewSectionCard>
      )}
    </div>
  )
}

interface ConversationDetailProps {
  conversation: ConvDetailType
  onBack: () => void
  onReviewGenerated: () => void
}

export function ConversationDetail({
  conversation,
  onBack,
  onReviewGenerated,
}: ConversationDetailProps) {
  const [generating, setGenerating] = useState(false)
  const date = conversation.started_at
    ? new Date(conversation.started_at).toLocaleString()
    : "—"
  const contact = conversation.contact_name || "Unknown"
  const scoreColor =
    conversation.final_close_score != null
      ? conversation.final_close_score >= 70
        ? "text-emerald-400"
        : conversation.final_close_score >= 40
          ? "text-amber-400"
          : "text-red-400"
      : ""

  async function handleGenerateReview() {
    setGenerating(true)
    try {
      await postConversationReview(conversation.id)
      onReviewGenerated()
    } catch (e) {
      console.error(e)
    } finally {
      setGenerating(false)
    }
  }

  return (
    <main className="flex-1 min-h-0 max-w-6xl w-full mx-auto p-4 overflow-y-auto flex flex-col">
      <div className="mb-4">
        <button
          type="button"
          onClick={onBack}
          className="text-sm text-slate-500 hover:text-white mb-3 flex items-center gap-1"
        >
          ← Back
        </button>
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold">
              <span dangerouslySetInnerHTML={{ __html: esc(contact) }} />
              {conversation.contact_company && (
                <span className="text-slate-500 font-normal">
                  {" "}
                  @ <span dangerouslySetInnerHTML={{ __html: esc(conversation.contact_company) }} />
                </span>
              )}
            </h2>
            <p className="text-xs text-slate-500">
              {date} · {conversation.mode} mode
            </p>
            {conversation.next_step && (
              <p className="text-sm text-cyan-400 mt-1 font-medium">
                →{" "}
                <span dangerouslySetInnerHTML={{ __html: esc(conversation.next_step) }} />
              </p>
            )}
          </div>
          <div className="flex items-center gap-3">
            {conversation.final_close_score != null && (
              <div className="text-right">
                <span className={cn("text-2xl font-black block", scoreColor)}>
                  {conversation.final_close_score}%
                </span>
                <span className="text-[10px] text-slate-500">Chance to close</span>
              </div>
            )}
            {!conversation.review && (
              <button
                type="button"
                onClick={handleGenerateReview}
                disabled={generating}
                className="px-3 py-1.5 text-xs rounded-lg bg-purple-600 hover:bg-purple-500 font-medium disabled:opacity-50"
              >
                {generating ? "Generating…" : "Generate Review"}
              </button>
            )}
          </div>
        </div>
      </div>

      {conversation.review && (
        <div className="mb-4">
          <h3 className="text-sm font-semibold text-purple-400 mb-2">
            📊 Post-Call Review
          </h3>
          <ReviewBlock review={conversation.review} />
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 max-h-[calc(100vh-14rem)] min-h-0">
        <div className="bg-slate-900 rounded-xl border border-slate-800 flex flex-col min-h-0">
          <h3 className="text-sm font-semibold text-slate-400 mb-2 px-3 pt-3 shrink-0">
            📝 Transcript ({(conversation.transcript || []).length} turns)
          </h3>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-1.5">
            {(conversation.transcript || []).map((t) => (
              <div key={t.turn} className="bg-slate-800/60 rounded-lg px-3 py-2">
                <span className="text-[10px] font-bold text-slate-500">
                  TURN {t.turn}
                </span>
                <p
                  className="text-[12px] text-slate-300 mt-0.5"
                  dangerouslySetInnerHTML={{ __html: esc(t.text) }}
                />
              </div>
            ))}
          </div>
        </div>

        <div className="bg-slate-900 rounded-xl border border-slate-800 flex flex-col min-h-0">
          <h3 className="text-sm font-semibold text-emerald-400 mb-2 px-3 pt-3 shrink-0">
            🎯 Coaching ({(conversation.coaching || []).length} tips)
          </h3>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-1.5">
            {(conversation.coaching || []).map((tip) => {
              const border = tip.is_custom
                ? "border-blue-900/40 bg-blue-950/30"
                : "border-emerald-900/40 bg-emerald-950/30"
              return (
                <div
                  key={`${tip.number}-${tip.turn}`}
                  className={cn("rounded-lg px-3 py-2 border", border)}
                >
                  <span
                    className={cn(
                      "text-[10px] font-bold",
                      tip.is_custom ? "text-blue-400" : "text-emerald-500"
                    )}
                  >
                    #{tip.number} · turn {tip.turn}
                  </span>
                  {tip.close_score >= 0 && (
                    <span className="text-[10px] text-slate-600 ml-2">
                      {tip.close_score}% · {tip.steps_to_close} steps
                    </span>
                  )}
                  <p
                    className="text-[12px] text-emerald-300 mt-0.5"
                    dangerouslySetInnerHTML={{ __html: esc(extractAdviceDisplay(tip.advice)) }}
                  />
                </div>
              )
            })}
          </div>
        </div>

        <div className="bg-slate-900 rounded-xl border border-slate-800 flex flex-col min-h-0">
          <h3 className="text-sm font-semibold text-amber-400 mb-2 px-3 pt-3 shrink-0">
            ⚠️ Hesitations{" "}
            <span className="text-[10px] font-normal text-slate-500">
              S · M · L
            </span>
          </h3>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-1.5">
            {conversation.hesitations && conversation.hesitations.length > 0 ? (
              conversation.hesitations.map((h, i) => {
                const resolved = h.status === "resolved"
                const sug = h.resolution_suggestion?.trim()
                return (
                  <div
                    key={i}
                    className="rounded-lg px-2 py-1.5 border border-slate-700/50"
                  >
                    <div
                      className={cn(
                        "flex items-center gap-1.5 flex-wrap text-[12px]",
                        resolved
                          ? "text-emerald-400 line-through opacity-60"
                          : "text-amber-300"
                      )}
                    >
                      <span>{resolved ? "✅" : "⚠️"}</span>
                      <RankBadge rank={h.rank} />
                      <span dangerouslySetInnerHTML={{ __html: esc(h.text) }} />
                    </div>
                    {sug && (
                      <div className="mt-1 pl-4 text-[11px] text-slate-400">
                        ↪ <span dangerouslySetInnerHTML={{ __html: esc(sug) }} />
                      </div>
                    )}
                  </div>
                )
              })
            ) : (
              <p className="text-slate-600 text-sm italic">None tracked</p>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
