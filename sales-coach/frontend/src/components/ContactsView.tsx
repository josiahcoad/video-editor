import { useState } from "react"
import ReactMarkdown from "react-markdown"
import type { ContactSummary } from "@/types/api"
import {
  createContact,
  deleteContact,
  fetchColdCallPrep,
  fetchContactDetail,
  updateContact,
} from "@/lib/api"
import { cn } from "@/lib/utils"
import { Globe, Loader2, Phone, Plus, Trash2 } from "lucide-react"

const STATUS_COLORS: Record<
  string,
  { bg: string; text: string }
> = {
  prospect: { bg: "bg-slate-700", text: "text-slate-300" },
  interested: { bg: "bg-blue-900/50", text: "text-blue-300" },
  demoing: { bg: "bg-purple-900/50", text: "text-purple-300" },
  reviewing_contract: { bg: "bg-amber-900/50", text: "text-amber-300" },
  closed_won: { bg: "bg-emerald-900/50", text: "text-emerald-300" },
  closed_lost: { bg: "bg-red-900/50", text: "text-red-400" },
}
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

function formatLastConvo(iso: string | null): string {
  if (!iso) return ""
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ""
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "2-digit",
  })
}

function StatusBadge({ status }: { status: string }) {
  const c = STATUS_COLORS[status] || STATUS_COLORS.prospect
  return (
    <span
      className={cn(
        "text-[10px] font-semibold px-2 py-0.5 rounded-full",
        c.bg,
        c.text
      )}
    >
      {STATUS_LABELS[status] || status}
    </span>
  )
}

const STATUS_OPTIONS = Object.keys(STATUS_LABELS)

interface ContactsViewProps {
  contacts: ContactSummary[]
  loading: boolean
  error: string | null
  onSelectContact: (id: number) => void
  onContactsReload?: () => void
}

export function ContactsView({
  contacts,
  loading,
  error,
  onSelectContact,
  onContactsReload,
}: ContactsViewProps) {
  const [showAddForm, setShowAddForm] = useState(false)
  const [addName, setAddName] = useState("")
  const [addCompany, setAddCompany] = useState("")
  const [addPhone, setAddPhone] = useState("")
  const [addStatus, setAddStatus] = useState("prospect")
  const [addError, setAddError] = useState<string | null>(null)
  const [addSubmitting, setAddSubmitting] = useState(false)
  const [deletingId, setDeletingId] = useState<number | null>(null)
  const [coldCallPrep, setColdCallPrep] = useState<{
    contactId: number
    name: string
    company: string | null
    content: string
  } | null>(null)
  const [coldCallPrepLoadingId, setColdCallPrepLoadingId] = useState<
    number | null
  >(null)
  const [savingPrepToNotes, setSavingPrepToNotes] = useState(false)
  const [savedPrepToNotes, setSavedPrepToNotes] = useState(false)

  const handleAddSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    const name = addName.trim()
    if (!name) {
      setAddError("Name is required")
      return
    }
    setAddError(null)
    setAddSubmitting(true)
    try {
      await createContact({
        name,
        company: addCompany.trim() || null,
        phone: addPhone.trim() || null,
        status: addStatus,
      })
      setAddName("")
      setAddCompany("")
      setAddPhone("")
      setAddStatus("prospect")
      setShowAddForm(false)
      onContactsReload?.()
    } catch (err) {
      setAddError(err instanceof Error ? err.message : "Failed to create contact")
    } finally {
      setAddSubmitting(false)
    }
  }

  const handleDelete = async (e: React.MouseEvent, id: number, name: string) => {
    e.stopPropagation()
    if (
      !window.confirm(
        `Delete "${name}"? This will remove this contact and all their conversations and notes.`
      )
    ) {
      return
    }
    setDeletingId(id)
    try {
      await deleteContact(id)
      onContactsReload?.()
    } finally {
      setDeletingId(null)
    }
  }

  if (loading) {
    return (
      <main className="flex-1 min-h-0 max-w-5xl w-full mx-auto p-4 overflow-y-auto">
        <p className="text-slate-600 italic text-sm">Loading…</p>
      </main>
    )
  }
  if (error) {
    return (
      <main className="flex-1 min-h-0 max-w-5xl w-full mx-auto p-4 overflow-y-auto">
        <p className="text-red-400 text-sm">{error}</p>
      </main>
    )
  }

  return (
    <main className="flex-1 min-h-0 max-w-5xl w-full mx-auto p-4 overflow-y-auto">
      <div className="space-y-3">
        <div className="flex items-center justify-between gap-2">
          <button
            type="button"
            onClick={() => setShowAddForm((v) => !v)}
            className="inline-flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-lg bg-slate-800 text-slate-200 hover:bg-slate-700 border border-slate-700"
          >
            <Plus className="w-4 h-4" />
            Add contact
          </button>
        </div>

        {showAddForm && (
          <form
            onSubmit={handleAddSubmit}
            className="bg-slate-900 border border-slate-700 rounded-xl p-4 space-y-3"
          >
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Name
              </label>
              <input
                type="text"
                value={addName}
                onChange={(e) => setAddName(e.target.value)}
                placeholder="Contact name"
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
                autoFocus
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Company (optional)
              </label>
              <input
                type="text"
                value={addCompany}
                onChange={(e) => setAddCompany(e.target.value)}
                placeholder="Company"
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Phone (optional)
              </label>
              <input
                type="tel"
                value={addPhone}
                onChange={(e) => setAddPhone(e.target.value)}
                placeholder="+1 555 123 4567"
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">
                Status
              </label>
              <select
                value={addStatus}
                onChange={(e) => setAddStatus(e.target.value)}
                className="w-full px-3 py-2 rounded-lg bg-slate-800 border border-slate-700 text-slate-200 text-sm focus:outline-none focus:ring-1 focus:ring-cyan-500"
              >
                {STATUS_OPTIONS.map((s) => (
                  <option key={s} value={s}>
                    {STATUS_LABELS[s] || s}
                  </option>
                ))}
              </select>
            </div>
            {addError && (
              <p className="text-red-400 text-sm">{addError}</p>
            )}
            <div className="flex gap-2">
              <button
                type="submit"
                disabled={addSubmitting}
                className="px-3 py-1.5 text-sm font-medium rounded-lg bg-cyan-700 text-white hover:bg-cyan-600 disabled:opacity-50"
              >
                {addSubmitting ? "Saving…" : "Save"}
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowAddForm(false)
                  setAddError(null)
                }}
                className="px-3 py-1.5 text-sm font-medium rounded-lg bg-slate-700 text-slate-300 hover:bg-slate-600"
              >
                Cancel
              </button>
            </div>
          </form>
        )}

        {contacts.length === 0 && !showAddForm && (
          <p className="text-slate-600 italic text-sm">
            No contacts yet. Click “Add contact” to create one.
          </p>
        )}

        {contacts.map((c) => {
          const score =
            c.latest_close_score != null ? `${c.latest_close_score}%` : ""
          const scoreColor =
            c.latest_close_score != null
              ? c.latest_close_score >= 70
                ? "text-emerald-400"
                : c.latest_close_score >= 40
                  ? "text-amber-400"
                  : "text-red-400"
              : ""
          const hesCount = c.open_hesitation_count || 0
          const isDeleting = deletingId === c.id
          return (
            <div
              key={c.id}
              role="button"
              tabIndex={0}
              onClick={() => onSelectContact(c.id)}
              onKeyDown={(e) =>
                (e.key === "Enter" || e.key === " ") && onSelectContact(c.id)
              }
              className="bg-slate-900 border border-slate-800 rounded-xl px-4 py-3 cursor-pointer hover:border-slate-600 transition"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-sm">
                        <span dangerouslySetInnerHTML={{ __html: esc(c.name) }} />
                      </span>
                      {c.company && (
                        <span className="text-xs text-slate-600">
                          @ <span dangerouslySetInnerHTML={{ __html: esc(c.company) }} />
                        </span>
                      )}
                      <StatusBadge status={c.status || "prospect"} />
                    </div>
                    {c.next_step && (
                      <p className="text-xs text-cyan-400 mt-1 flex items-center gap-1">
                        → <span dangerouslySetInnerHTML={{ __html: esc(c.next_step) }} />
                      </p>
                    )}
                    {c.last_conversation_at && (
                      <p className="text-xs text-slate-500 mt-1">
                        Last convo: {formatLastConvo(c.last_conversation_at)}
                      </p>
                    )}
                  </div>
                </div>
                <div className="text-right flex items-center gap-3">
                  {hesCount > 0 && (
                    <span className="text-[11px] text-amber-400 font-medium">
                      {hesCount} open
                    </span>
                  )}
                  {score && (
                    <span className={cn("font-bold text-lg", scoreColor)}>
                      {score}
                    </span>
                  )}
                  <span className="text-[10px] text-slate-600">
                    {c.conversation_count || 0} calls
                  </span>
                  <button
                    type="button"
                    onClick={async (e) => {
                      e.stopPropagation()
                      setColdCallPrepLoadingId(c.id)
                      try {
                        const res = await fetchColdCallPrep(
                          c.name,
                          c.company ?? null
                        )
                        setColdCallPrep({
                          contactId: c.id,
                          name: c.name,
                          company: c.company ?? null,
                          content: res.content,
                        })
                        setSavedPrepToNotes(false)
                      } catch (err) {
                        alert(
                          err instanceof Error ? err.message : "Cold call prep failed"
                        )
                      } finally {
                        setColdCallPrepLoadingId(null)
                      }
                    }}
                    disabled={coldCallPrepLoadingId === c.id}
                    className="p-1.5 rounded text-slate-500 hover:text-sky-400 hover:bg-slate-800 disabled:opacity-50"
                    title="Web search: cold call prep for this business"
                    aria-label={`Cold call prep for ${c.name}`}
                  >
                    {coldCallPrepLoadingId === c.id ? (
                      <Loader2 className="w-4 h-4 animate-spin" />
                    ) : (
                      <Globe className="w-4 h-4" />
                    )}
                  </button>
                  {c.phone && (
                    <a
                      href={`tel:${c.phone}`}
                      onClick={(e) => e.stopPropagation()}
                      className="p-1.5 rounded text-slate-500 hover:text-emerald-400 hover:bg-slate-800"
                      title={`Call ${c.phone}`}
                      aria-label={`Call ${c.name}`}
                    >
                      <Phone className="w-4 h-4" />
                    </a>
                  )}
                  <button
                    type="button"
                    onClick={(e) => handleDelete(e, c.id, c.name)}
                    disabled={isDeleting}
                    className="p-1.5 rounded text-slate-500 hover:text-red-400 hover:bg-slate-800 disabled:opacity-50"
                    title="Delete contact"
                    aria-label={`Delete ${c.name}`}
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {coldCallPrep && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60"
          onClick={() => setColdCallPrep(null)}
          role="dialog"
          aria-modal="true"
          aria-labelledby="cold-call-prep-title"
        >
          <div
            className="bg-slate-900 border border-slate-700 rounded-xl shadow-xl max-w-lg w-full max-h-[80vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
              <h3
                id="cold-call-prep-title"
                className="text-base font-semibold text-slate-200"
              >
                Cold call prep: {coldCallPrep.name}
                {coldCallPrep.company && (
                  <span className="text-slate-500 font-normal">
                    {" "}
                    @ {coldCallPrep.company}
                  </span>
                )}
              </h3>
              <button
                type="button"
                onClick={() => setColdCallPrep(null)}
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
                {coldCallPrep.content}
              </ReactMarkdown>
            </div>
            <div className="px-4 py-3 border-t border-slate-700 flex justify-end gap-2">
              <button
                type="button"
                onClick={async () => {
                  if (savingPrepToNotes || savedPrepToNotes) return
                  setSavingPrepToNotes(true)
                  try {
                    const contact = await fetchContactDetail(
                      coldCallPrep.contactId
                    )
                    const existing = (contact.notes ?? "").trim()
                    const separator =
                      existing ? "\n\n--- Cold call prep ---\n\n" : ""
                    await updateContact(coldCallPrep.contactId, {
                      notes: existing + separator + coldCallPrep.content,
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
                onClick={() => setColdCallPrep(null)}
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
