import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import type { ContactOption } from "@/types/api"
import { createContact, fetchContactDetail, fetchPrepare } from "@/lib/api"

const STATUS_OPTIONS = [
  "prospect",
  "interested",
  "demoing",
  "reviewing_contract",
] as const

interface ContextBarProps {
  contacts: ContactOption[]
  contactId: string
  onContactIdChange: (id: string) => void
  disabled?: boolean
  onContactsReload: () => void
}

export function ContextBar({
  contacts,
  contactId,
  onContactIdChange,
  disabled,
  onContactsReload,
}: ContextBarProps) {
  const [showNew, setShowNew] = useState(false)
  const [newName, setNewName] = useState("")
  const [newCompany, setNewCompany] = useState("")
  const [newStatus, setNewStatus] = useState<string>("prospect")
  const [saving, setSaving] = useState(false)
  const [notes, setNotes] = useState<string | null>(null)
  const [notesLoading, setNotesLoading] = useState(false)
  const [prepareContent, setPrepareContent] = useState<string | null>(null)
  const [prepareLoading, setPrepareLoading] = useState(false)

  useEffect(() => {
    if (!contactId) {
      setNotes(null)
      setPrepareContent(null)
      return
    }
    let cancelled = false
    setNotesLoading(true)
    fetchContactDetail(Number(contactId))
      .then((c) => {
        if (!cancelled) setNotes(c.notes ?? null)
      })
      .catch(() => {
        if (!cancelled) setNotes(null)
      })
      .finally(() => {
        if (!cancelled) setNotesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [contactId])

  async function handleSaveNew() {
    const name = newName.trim()
    if (!name) return
    setSaving(true)
    try {
      const c = await createContact({
        name,
        company: newCompany.trim() || null,
        status: newStatus,
      })
      await onContactsReload()
      onContactIdChange(String(c.id))
      setShowNew(false)
      setNewName("")
      setNewCompany("")
    } catch (e) {
      console.error(e)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="shrink-0 max-w-7xl w-full mx-auto px-4 pt-3 pb-0">
      <details open className="bg-slate-900 rounded-xl border border-slate-800">
        <summary className="px-4 py-2 cursor-pointer text-sm font-semibold text-slate-400 select-none">
          📋 Pre-Call Context
          <span className="text-xs font-normal text-slate-600 ml-2">
            contact notes
          </span>
        </summary>
        <div className="px-4 pb-3 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <select
              value={contactId}
              onChange={(e) => onContactIdChange(e.target.value)}
              disabled={disabled}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-emerald-600 disabled:opacity-50"
            >
              <option value="">No contact selected</option>
              {contacts.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.company ? `${c.name} (${c.company})` : c.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setShowNew(true)}
              className="px-2.5 py-1.5 text-[11px] rounded-lg bg-slate-700 hover:bg-slate-600 font-medium"
            >
              + New
            </button>
            {showNew && (
              <div className="flex items-center gap-2 flex-wrap">
                <input
                  type="text"
                  placeholder="Name"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-slate-200 w-32 focus:outline-none focus:border-emerald-600"
                />
                <input
                  type="text"
                  placeholder="Company"
                  value={newCompany}
                  onChange={(e) => setNewCompany(e.target.value)}
                  className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-sm text-slate-200 w-32 focus:outline-none focus:border-emerald-600"
                />
                <select
                  value={newStatus}
                  onChange={(e) => setNewStatus(e.target.value)}
                  className="bg-slate-800 border border-slate-700 rounded-lg px-2 py-1.5 text-[11px] text-slate-300 focus:outline-none"
                >
                  {STATUS_OPTIONS.map((s) => (
                    <option key={s} value={s}>
                      {s === "reviewing_contract" ? "Reviewing Contract" : s}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  onClick={handleSaveNew}
                  disabled={saving}
                  className="px-2.5 py-1.5 text-[11px] rounded-lg bg-emerald-700 hover:bg-emerald-600 font-medium disabled:opacity-50"
                >
                  Save
                </button>
                <button
                  type="button"
                  onClick={() => setShowNew(false)}
                  className="px-2 py-1.5 text-[11px] text-slate-500 hover:text-slate-300"
                >
                  Cancel
                </button>
              </div>
            )}
          </div>
          <div className="min-h-16">
            {!contactId ? (
              <p className="text-sm text-slate-500 italic py-2">
                Select a contact to see their notes
              </p>
            ) : notesLoading ? (
              <p className="text-sm text-slate-500 italic py-2">Loading notes…</p>
            ) : notes ? (
              <div className="text-sm text-slate-300 whitespace-pre-wrap bg-slate-800 border border-slate-700 rounded-lg px-3 py-2">
                {notes}
              </div>
            ) : (
              <p className="text-sm text-slate-500 italic py-2">
                No notes for this contact
              </p>
            )}
          </div>
          {contactId && (
            <div className="pt-2 space-y-2">
              <button
                type="button"
                onClick={async () => {
                  setPrepareLoading(true)
                  setPrepareContent(null)
                  try {
                    const res = await fetchPrepare(Number(contactId))
                    setPrepareContent(res.content)
                  } catch (e) {
                    alert(e instanceof Error ? e.message : "Failed to generate preparation")
                  } finally {
                    setPrepareLoading(false)
                  }
                }}
                disabled={prepareLoading}
                className="px-3 py-1.5 text-sm rounded-lg bg-emerald-700 hover:bg-emerald-600 font-medium disabled:opacity-50 text-white"
              >
                {prepareLoading ? "Generating…" : "Help me prepare"}
              </button>
              {prepareContent !== null && (
                <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 max-h-64 overflow-y-auto text-sm text-slate-300 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-slate-200 [&_h1]:mt-2 [&_h1]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-slate-200 [&_h2]:mt-2 [&_h2]:mb-1 [&_p]:my-1 [&_ul]:my-1.5 [&_ul]:pl-5 [&_li]:my-0.5 [&_strong]:font-semibold [&_strong]:text-slate-200">
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
                    {prepareContent}
                  </ReactMarkdown>
                  <button
                    type="button"
                    onClick={() => setPrepareContent(null)}
                    className="mt-2 text-xs text-slate-500 hover:text-slate-400"
                  >
                    Clear
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </details>
    </div>
  )
}
