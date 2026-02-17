import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import type { ContactDetail as ContactDetailType } from "@/types/api"
import {
  deleteConversation,
  fetchColdCallPrep,
  fetchContactDetail,
  updateContact,
} from "@/lib/api"
import { cn } from "@/lib/utils"
import { Trash2 } from "lucide-react"

const STATUS_OPTIONS = [
  "prospect",
  "interested",
  "demoing",
  "reviewing_contract",
  "closed_won",
  "closed_lost",
] as const
const STATUS_LABELS: Record<string, string> = {
  prospect: "Prospect",
  interested: "Interested",
  demoing: "Demoing",
  reviewing_contract: "Reviewing Contract",
  closed_won: "Closed Won",
  closed_lost: "Closed Lost",
}

function esc(s: string): string {
  if (!s) return ""
  const div = document.createElement("div")
  div.textContent = s
  return div.innerHTML
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

interface ContactDetailProps {
  contact: ContactDetailType
  onBack: () => void
  onSelectConversation: (convId: number, contactId: number) => void
  onStatusChange: () => void
  onStartConversation?: (contactId: number) => void
}

export function ContactDetail({
  contact,
  onBack,
  onSelectConversation,
  onStatusChange,
  onStartConversation,
}: ContactDetailProps) {
  const [deletingConvId, setDeletingConvId] = useState<number | null>(null)
  const [notesValue, setNotesValue] = useState(contact.notes ?? "")
  const [notesSaving, setNotesSaving] = useState(false)
  const [researchContent, setResearchContent] = useState<string | null>(null)
  const [researchLoading, setResearchLoading] = useState(false)
  const [savingResearch, setSavingResearch] = useState(false)
  const [savedResearch, setSavedResearch] = useState(false)

  useEffect(() => {
    setNotesValue(contact.notes ?? "")
  }, [contact.id, contact.notes])

  async function handleStatusChange(value: string) {
    await updateContact(contact.id, { status: value })
    onStatusChange()
  }

  async function handleDeleteConversation(
    e: React.MouseEvent,
    convId: number,
    dateLabel: string
  ) {
    e.stopPropagation()
    if (
      !window.confirm(
        `Delete conversation from ${dateLabel}? This will remove the transcript, coaching, and hesitations for this conversation.`
      )
    )
      return
    setDeletingConvId(convId)
    try {
      await deleteConversation(convId)
      onStatusChange()
    } finally {
      setDeletingConvId(null)
    }
  }

  async function handleNotesBlur() {
    const value = notesValue.trim()
    if (value === (contact.notes ?? "").trim()) return
    setNotesSaving(true)
    try {
      await updateContact(contact.id, { notes: value || "" })
      onStatusChange()
    } finally {
      setNotesSaving(false)
    }
  }

  const dateStr = (s: string | null) =>
    s
      ? new Date(s).toLocaleDateString("en-US", {
          month: "short",
          day: "numeric",
          hour: "numeric",
          minute: "2-digit",
        })
      : "—"

  return (
    <main className="flex-1 min-h-0 max-w-6xl w-full mx-auto p-4 overflow-y-auto flex flex-col">
      <div className="mb-4">
        <button
          type="button"
          onClick={onBack}
          className="text-sm text-slate-500 hover:text-white mb-3 flex items-center gap-1"
        >
          ← Back to contacts
        </button>
        <div className="flex items-center justify-between">
          <div>
            <div className="flex items-center gap-3 flex-wrap">
              <h2 className="text-lg font-bold">
                <span dangerouslySetInnerHTML={{ __html: esc(contact.name) }} />
              </h2>
              {contact.company && (
                <span className="text-slate-500">
                  @ <span dangerouslySetInnerHTML={{ __html: esc(contact.company) }} />
                </span>
              )}
              <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-700 text-slate-300">
                {STATUS_LABELS[contact.status] || contact.status}
              </span>
              <select
                value={contact.status}
                onChange={(e) => handleStatusChange(e.target.value)}
                className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1 text-[11px] text-slate-300 focus:outline-none"
              >
                {STATUS_OPTIONS.map((s) => (
                  <option key={s} value={s}>
                    {STATUS_LABELS[s]}
                  </option>
                ))}
              </select>
            </div>
            {contact.next_step && (
              <p className="text-sm text-cyan-400 mt-1 font-medium">
                → Next:{" "}
                <span dangerouslySetInnerHTML={{ __html: esc(contact.next_step) }} />
              </p>
            )}
            <p className="text-sm text-slate-400 mt-1 flex items-center gap-2 flex-wrap">
              {contact.phone && (
                <>
                  Phone:{" "}
                  <a
                    href={`tel:${contact.phone}`}
                    onClick={(e) => e.stopPropagation()}
                    className="text-cyan-400 hover:text-cyan-300 underline"
                  >
                    {contact.phone}
                  </a>
                </>
              )}
              <button
                  type="button"
                  onClick={async () => {
                    setResearchLoading(true)
                    setResearchContent(null)
                    setSavedResearch(false)
                    try {
                      const res = await fetchColdCallPrep(
                        contact.name,
                        contact.company ?? null
                      )
                      setResearchContent(res.content)
                    } catch (e) {
                      alert(
                        e instanceof Error ? e.message : "Research failed"
                      )
                    } finally {
                      setResearchLoading(false)
                    }
                  }}
                  disabled={researchLoading}
                  className="px-2.5 py-1 text-xs font-medium rounded-lg bg-sky-700 text-white hover:bg-sky-600 disabled:opacity-50"
                >
                {researchLoading ? "Researching…" : "Research"}
              </button>
            </p>
          </div>
        </div>
        <div className="mt-2">
          <label className="block text-xs font-medium text-slate-500 mb-1">
            Notes
            {notesSaving && (
              <span className="ml-2 text-slate-600 italic">Saving…</span>
            )}
          </label>
          <textarea
            value={notesValue}
            onChange={(e) => setNotesValue(e.target.value)}
            onBlur={handleNotesBlur}
            placeholder="Add notes about this contact…"
            rows={4}
            className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500 resize-y min-h-[80px]"
          />
        </div>
        {(contact.research ?? "").trim() && (
          <div className="mt-3">
            <label className="block text-xs font-medium text-slate-500 mb-1">
              Research
            </label>
            <div className="rounded-lg bg-slate-800 border border-slate-700 p-3 text-sm text-slate-300 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-slate-200 [&_h1]:mt-2 [&_h1]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-slate-200 [&_h2]:mt-2 [&_h2]:mb-1 [&_p]:my-1 [&_ul]:my-1.5 [&_ul]:pl-5 [&_li]:my-0.5 [&_strong]:font-semibold [&_strong]:text-slate-200">
              <ReactMarkdown
                components={{
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-cyan-400 hover:text-cyan-300 hover:underline"
                    >
                      {children}
                    </a>
                  ),
                }}
              >
                {contact.research ?? ""}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      {contact.open_hesitations && contact.open_hesitations.length > 0 && (
        <div className="mb-4 bg-amber-950/20 border border-amber-900/30 rounded-xl p-3">
          <h3 className="text-sm font-semibold text-amber-400 mb-2">
            ⚠️ Open Hesitations ({contact.open_hesitations.length}){" "}
            <span className="text-[10px] font-normal text-slate-500">
              S=minor · M=blocking · L=hard stop
            </span>
          </h3>
          <div className="space-y-2">
            {contact.open_hesitations.map((h, i) => {
              const sug = h.resolution_suggestion?.trim()
              return (
                <div
                  key={i}
                  className="rounded-lg px-2 py-1.5 border border-amber-900/30"
                >
                  <div className="flex items-center gap-1.5 flex-wrap text-[12px] text-amber-300">
                    <span>⚠️</span>
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
            })}
          </div>
        </div>
      )}

      <div className="space-y-3">
        <div className="flex items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-slate-400">
            📞 Conversations ({contact.conversations?.length ?? 0})
          </h3>
          {onStartConversation && (
            <button
              type="button"
              onClick={() => onStartConversation(contact.id)}
              className="px-3 py-1.5 text-xs font-medium rounded-lg bg-cyan-700 text-white hover:bg-cyan-600"
            >
              New Conversation
            </button>
          )}
        </div>
        {!contact.conversations?.length ? (
          <p className="text-slate-600 italic text-sm">No conversations yet.</p>
        ) : (
          contact.conversations.map((conv) => {
            const score =
              conv.final_close_score != null ? `${conv.final_close_score}%` : ""
            const scoreColor =
              conv.final_close_score != null
                ? conv.final_close_score >= 70
                  ? "text-emerald-400"
                  : conv.final_close_score >= 40
                    ? "text-amber-400"
                    : "text-red-400"
                : ""
            return (
              <div
                key={conv.id}
                role="button"
                tabIndex={0}
                onClick={() => onSelectConversation(conv.id, contact.id)}
                onKeyDown={(e) =>
                  (e.key === "Enter" || e.key === " ") &&
                  onSelectConversation(conv.id, contact.id)
                }
                className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 cursor-pointer hover:border-slate-600 transition"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-slate-500">
                      {dateStr(conv.started_at)}
                    </span>
                    {conv.mode === "test" && (
                      <span className="text-[10px] bg-blue-900/40 text-blue-400 px-1.5 py-0.5 rounded">
                        test
                      </span>
                    )}
                    {conv.review_generated_at && "📊"}
                    {conv.next_step && (
                      <span className="text-[11px] text-cyan-400">
                        →{" "}
                        <span
                          dangerouslySetInnerHTML={{
                            __html: esc(conv.next_step),
                          }}
                        />
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {score && (
                      <span className={cn("font-bold", scoreColor)}>
                        {score}
                      </span>
                    )}
                    {conv.final_steps_to_close != null && (
                      <span className="text-[10px] text-slate-600">
                        {conv.final_steps_to_close} steps
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={(e) =>
                        handleDeleteConversation(
                          e,
                          conv.id,
                          dateStr(conv.started_at)
                        )
                      }
                      disabled={deletingConvId === conv.id}
                      className="p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-slate-800 disabled:opacity-50"
                      title="Delete conversation"
                      aria-label="Delete conversation"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>

      {researchContent !== null && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60"
          onClick={() => setResearchContent(null)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="research-modal-title"
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
              <h3
                id="research-modal-title"
                className="text-base font-semibold text-slate-200"
              >
                Cold call prep: {contact.name}
                {contact.company && (
                  <span className="text-slate-500 font-normal">
                    {" "}
                    @ {contact.company}
                  </span>
                )}
              </h3>
              <button
                type="button"
                onClick={() => setResearchContent(null)}
                className="p-1.5 rounded text-slate-400 hover:text-slate-200 hover:bg-slate-800"
                aria-label="Close"
              >
                ×
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1 text-sm text-slate-300 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-slate-200 [&_h1]:mt-3 [&_h1]:mb-2 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-slate-200 [&_h2]:mt-3 [&_h2]:mb-1 [&_p]:my-1.5 [&_ul]:my-2 [&_ul]:pl-5 [&_li]:my-0.5 [&_strong]:font-semibold [&_strong]:text-slate-200">
              <ReactMarkdown
                components={{
                  a: ({ href, children }) => (
                    <a
                      href={href}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-cyan-400 hover:text-cyan-300 hover:underline"
                    >
                      {children}
                    </a>
                  ),
                }}
              >
                {researchContent}
              </ReactMarkdown>
            </div>
            <div className="px-4 py-3 border-t border-slate-700 flex justify-end gap-2">
              <button
                type="button"
                onClick={async () => {
                  if (savingPrepToNotes || savedPrepToNotes) return
                  setSavingPrepToNotes(true)
                  try {
                    const c = await fetchContactDetail(contact.id)
                    const existing = (c.notes ?? "").trim()
                    const separator =
                      existing ? "\n\n--- Cold call prep ---\n\n" : ""
                    await updateContact(contact.id, {
                      notes: existing + separator + researchContent,
                    })
                    setSavedPrepToNotes(true)
                  } catch (e) {
                    alert(
                      e instanceof Error ? e.message : "Failed to save to notes"
                    )
                  } finally {
                    setSavingPrepToNotes(false)
                  }
                }}
                disabled={savingPrepToNotes || savedPrepToNotes}
                className="px-3 py-1.5 text-sm font-medium rounded-lg bg-cyan-700 text-white hover:bg-cyan-600 disabled:opacity-50 disabled:bg-slate-700"
              >
                {savingPrepToNotes
                  ? "Saving…"
                  : savedPrepToNotes
                    ? "Saved to notes"
                    : "Save to notes"}
              </button>
              <button
                type="button"
                onClick={() => setResearchContent(null)}
                className="px-3 py-1.5 text-sm font-medium rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  )
}
