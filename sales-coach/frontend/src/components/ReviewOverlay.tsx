import type { Review } from "@/types/api"
import { cn } from "@/lib/utils"
import type { ReactNode } from "react"

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

interface ReviewOverlayProps {
  open: boolean
  onClose: () => void
  review: Review | null
  loading: boolean
}

export function ReviewOverlay({
  open,
  onClose,
  review,
  loading,
}: ReviewOverlayProps) {
  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-labelledby="review-title"
    >
      <div className="bg-slate-900 border border-slate-700 rounded-2xl max-w-2xl w-full max-h-[85vh] overflow-y-auto p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 id="review-title" className="text-lg font-bold text-white">
            📊 Post-Call Review
          </h2>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-500 hover:text-white text-xl leading-none"
            aria-label="Close"
          >
            &times;
          </button>
        </div>
        <div>
          {loading && (
            <div className="flex items-center gap-3 text-slate-400">
              <span className="flex gap-0.5">
                {[0, 1, 2].map((i) => (
                  <span
                    key={i}
                    className="w-2 h-2 rounded-full bg-purple-500 animate-pulse-dot"
                    style={{ animationDelay: `${i * 0.15}s` }}
                  />
                ))}
              </span>
              Analyzing your call…
            </div>
          )}
          {!loading && !review && (
            <p className="text-slate-500 italic">Generating review…</p>
          )}
          {!loading && review && <ReviewContent review={review} esc={esc} />}
        </div>
      </div>
    </div>
  )
}

/** Section card for overlay review */
function OverlaySectionCard({
  title,
  titleClassName,
  children,
}: {
  title: string
  titleClassName?: string
  children: ReactNode
}) {
  return (
    <div className="rounded-xl border border-slate-700/60 bg-slate-800/50 p-4">
      <h3 className={cn("text-sm font-semibold mb-3", titleClassName)}>{title}</h3>
      {children}
    </div>
  )
}

function ReviewContent({ review, esc }: { review: Review; esc: (s: string) => string }) {
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
      {/* Score hero card */}
      <div className="rounded-xl border border-slate-700/60 bg-slate-800/50 p-4">
        <div className="flex items-center gap-3">
          <span className={cn("text-3xl font-black", scoreColor)}>
            {score != null ? `${score}/10` : "—/10"}
          </span>
          <span className="text-sm text-slate-400">Overall effectiveness</span>
        </div>
      </div>

      <OverlaySectionCard title="✅ What went well" titleClassName="text-emerald-400">
        <div className="space-y-2">
          {wentWell.length === 0 && <p className="text-[13px] text-slate-600">None</p>}
          {wentWell.map((w, i) => (
            <div
              key={i}
              className="rounded-lg border-l-2 border-emerald-700 bg-slate-800/40 py-2 px-3 text-[13px] text-slate-300"
              dangerouslySetInnerHTML={{ __html: esc(w) }}
            />
          ))}
        </div>
      </OverlaySectionCard>

      <OverlaySectionCard title="📈 Areas to improve" titleClassName="text-amber-400">
        <div className="space-y-2">
          {improve.length === 0 && <p className="text-[13px] text-slate-600">None</p>}
          {improve.map((w, i) => (
            <div
              key={i}
              className="rounded-lg border-l-2 border-amber-700 bg-slate-800/40 py-2 px-3 text-[13px] text-slate-300"
              dangerouslySetInnerHTML={{ __html: esc(w) }}
            />
          ))}
        </div>
      </OverlaySectionCard>

      {review.hesitations_summary && (
        <OverlaySectionCard title="🔍 Hesitations" titleClassName="text-purple-400">
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-slate-500 uppercase">Uncovered</p>
              {uncovered.length === 0 && <p className="text-[12px] text-slate-600">None</p>}
              {uncovered.map((h, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-slate-800/40 py-2 px-3 text-[12px] text-slate-300"
                  dangerouslySetInnerHTML={{ __html: esc(h) }}
                />
              ))}
            </div>
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-emerald-600 uppercase">Handled</p>
              {handled.length === 0 && <p className="text-[12px] text-slate-600">None</p>}
              {handled.map((h, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-emerald-950/30 border border-emerald-900/40 py-2 px-3 text-[12px] text-emerald-300"
                  dangerouslySetInnerHTML={{ __html: esc(h) }}
                />
              ))}
            </div>
            <div className="space-y-2">
              <p className="text-[10px] font-bold text-red-500 uppercase">Remaining</p>
              {remaining.length === 0 && <p className="text-[12px] text-slate-600">None</p>}
              {remaining.map((h, i) => (
                <div
                  key={i}
                  className="rounded-lg bg-red-950/20 border border-red-900/40 py-2 px-3 text-[12px] text-red-300"
                  dangerouslySetInnerHTML={{ __html: esc(h) }}
                />
              ))}
            </div>
          </div>
        </OverlaySectionCard>
      )}

      {review.key_moment && (
        <OverlaySectionCard title="💡 Key moment" titleClassName="text-blue-400">
          <p
            className="text-[13px] text-slate-300 rounded-lg bg-blue-950/30 px-3 py-2 border border-blue-900/30"
            dangerouslySetInnerHTML={{ __html: esc(review.key_moment) }}
          />
        </OverlaySectionCard>
      )}

      <OverlaySectionCard title="📋 Follow-ups" titleClassName="text-cyan-400">
        <div className="space-y-2">
          {followUps.length === 0 && <p className="text-[13px] text-slate-600">None</p>}
          {followUps.map((f, i) => (
            <div
              key={i}
              className="flex gap-2 rounded-lg bg-slate-800/40 py-2 px-3 text-[13px] text-slate-300"
            >
              <span className="text-cyan-600 font-bold shrink-0">{i + 1}.</span>
              <span dangerouslySetInnerHTML={{ __html: esc(f) }} />
            </div>
          ))}
        </div>
      </OverlaySectionCard>

      {review.next_step && (
        <OverlaySectionCard title="→ Next step" titleClassName="text-cyan-400">
          <p
            className="text-[13px] text-cyan-300 rounded-lg bg-cyan-950/30 px-3 py-2 border border-cyan-900/30 font-medium"
            dangerouslySetInnerHTML={{ __html: esc(review.next_step) }}
          />
        </OverlaySectionCard>
      )}

      {review.next_call_prep && (
        <OverlaySectionCard title="🗓️ Next call prep" titleClassName="text-slate-400">
          <p
            className="text-[13px] text-slate-300 rounded-lg bg-slate-800/40 px-3 py-2"
            dangerouslySetInnerHTML={{
              __html: formatParagraphWithBreaks(review.next_call_prep),
            }}
          />
        </OverlaySectionCard>
      )}

      {review.suggested_status && (
        <div
          className="text-[11px] text-slate-500 rounded-lg bg-slate-800/30 px-3 py-2"
          dangerouslySetInnerHTML={{
            __html: `Pipeline: ${esc(review.suggested_status)}`,
          }}
        />
      )}
    </div>
  )
}
