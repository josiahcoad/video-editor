import { useEffect, useState } from "react"
import ReactMarkdown from "react-markdown"
import { Phone } from "lucide-react"
import type { ContactOption } from "@/types/api"
import {
  createContact,
  fetchContactDetail,
  fetchConversation,
  fetchPrepareStream,
  updateConversationPrepNotes,
} from "@/lib/api"

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
  /** When on /live/:conversationId, pass so we can load/save prep from the conversation. */
  conversationId?: string | null
  conversationContactLoading?: boolean
  conversationHasNoContact?: boolean
}

export function ContextBar({
  contacts,
  contactId,
  onContactIdChange,
  disabled,
  onContactsReload,
  conversationId,
  conversationContactLoading,
  conversationHasNoContact,
}: ContextBarProps) {
  const [showNew, setShowNew] = useState(false)
  const [newName, setNewName] = useState("")
  const [newCompany, setNewCompany] = useState("")
  const [newStatus, setNewStatus] = useState<string>("prospect")
  const [saving, setSaving] = useState(false)
  const [notes, setNotes] = useState<string | null>(null)
  const [research, setResearch] = useState<string | null>(null)
  const [notesLoading, setNotesLoading] = useState(false)
  const [prepareContent, setPrepareContent] = useState<string | null>(null)
  const [prepareLoading, setPrepareLoading] = useState(false)

  useEffect(() => {
    if (!contactId) {
      setNotes(null)
      setResearch(null)
      setPrepareContent(null)
      return
    }
    let cancelled = false
    setNotesLoading(true)
    fetchContactDetail(Number(contactId))
      .then((c) => {
        if (!cancelled) {
          setNotes(c.notes ?? null)
          setResearch(c.research?.trim() ?? null)
        }
      })
      .catch(() => {
        if (!cancelled) setNotes(null)
        if (!cancelled) setResearch(null)
      })
      .finally(() => {
        if (!cancelled) setNotesLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [contactId])

  // When we're on a conversation page (/live/:id): load existing prep or stream and save. Requires both contactId and conversationId.
  useEffect(() => {
    if (!contactId || !conversationId) return
    const convId = Number(conversationId)
    if (Number.isNaN(convId)) return

    let cancelled = false

    async function loadOrGenerate() {
      try {
        const conv = await fetchConversation(convId)
        if (cancelled) return
        const existing = conv.prep_notes
        if (existing?.trim()) {
          setPrepareContent(existing.trim())
          return
        }
        setPrepareLoading(true)
        setPrepareContent("")
        const res = await fetchPrepareStream(Number(contactId))
        if (cancelled) return
        const reader = res.body?.getReader()
        if (!reader) {
          setPrepareContent("Stream not available")
          setPrepareLoading(false)
          return
        }
        const decoder = new TextDecoder()
        let text = ""
        try {
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            if (cancelled) return
            text += decoder.decode(value, { stream: true })
            setPrepareContent(text)
          }
        } finally {
          reader.releaseLock()
        }
        if (!cancelled && text) {
          await updateConversationPrepNotes(convId, text)
        }
      } catch (e) {
        if (!cancelled) {
          setPrepareContent(
            e instanceof Error ? e.message : "Failed to load or generate preparation"
          )
        }
      } finally {
        if (!cancelled) setPrepareLoading(false)
      }
    }

    loadOrGenerate()
    return () => {
      cancelled = true
    }
  }, [contactId, conversationId])

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

  const selectedContact = contactId
    ? contacts.find((c) => String(c.id) === contactId)
    : null
  const contactPhone = selectedContact?.phone?.trim() ?? null

  return (
    <div className="shrink-0 max-w-7xl w-full mx-auto px-4 pt-3 pb-0">
      <details open className="bg-slate-900 rounded-xl border border-slate-800">
        <summary className="px-4 py-2 cursor-pointer text-sm font-semibold text-slate-400 select-none flex items-center gap-2">
          📋 Contact Details
          <span className="text-xs font-normal text-slate-600">
            notes · research · prep
          </span>
          {contactPhone ? (
            <a
              href={`tel:${contactPhone}`}
              onClick={(e) => e.stopPropagation()}
              className="ml-auto p-1.5 rounded-lg text-slate-400 hover:text-emerald-400 hover:bg-slate-800"
              title={`Call ${contactPhone}`}
              aria-label={`Call ${contactPhone}`}
            >
              <Phone className="w-4 h-4" />
            </a>
          ) : contactId ? (
            <span
              className="ml-auto p-1.5 rounded-lg text-slate-600 cursor-default"
              title="No phone number"
              aria-label="No phone number"
            >
              <Phone className="w-4 h-4" />
            </span>
          ) : null}
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
          <div className="min-h-16 grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Left column: Notes + Research */}
            <div className="space-y-2">
              {!contactId ? (
                <p className="text-sm text-slate-500 italic py-2">
                  {conversationContactLoading
                    ? "Loading contact…"
                    : conversationHasNoContact
                      ? "This conversation has no contact linked."
                      : "Select a contact to see their notes"}
                </p>
              ) : notesLoading ? (
                <p className="text-sm text-slate-500 italic py-2">Loading notes…</p>
              ) : (
                <>
                  {notes ? (
                    <div>
                      <p className="text-[11px] font-medium text-slate-500 mb-1">
                        Notes
                      </p>
                      <div className="text-sm text-slate-300 whitespace-pre-wrap bg-slate-800 border border-slate-700 rounded-lg px-3 py-2">
                        {notes}
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500 italic py-2">
                      No notes for this contact
                    </p>
                  )}
                  {research && (
                    <div>
                      <p className="text-[11px] font-medium text-slate-500 mb-1">
                        Research
                      </p>
                      <div className="text-sm text-slate-300 bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 max-h-64 overflow-y-auto [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-slate-200 [&_h1]:mt-2 [&_h1]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-slate-200 [&_h2]:mt-2 [&_h2]:mb-1 [&_p]:my-1 [&_ul]:my-1.5 [&_ul]:pl-5 [&_li]:my-0.5 [&_strong]:font-semibold [&_strong]:text-slate-200">
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
                          {research}
                        </ReactMarkdown>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
            {/* Right column: Prep (always show so layout is two columns) */}
            <div className="space-y-2">
              <p className="text-[11px] font-medium text-slate-500 mb-1">
                Prep
              </p>
              <div className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 min-h-32 max-h-64 overflow-y-auto text-sm text-slate-300 [&_h1]:text-base [&_h1]:font-semibold [&_h1]:text-slate-200 [&_h1]:mt-2 [&_h1]:mb-1 [&_h2]:text-sm [&_h2]:font-semibold [&_h2]:text-slate-200 [&_h2]:mt-2 [&_h2]:mb-1 [&_p]:my-1 [&_ul]:my-1.5 [&_ul]:pl-5 [&_li]:my-0.5 [&_strong]:font-semibold [&_strong]:text-slate-200">
                {prepareContent != null && prepareContent !== "" ? (
                  <>
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
                  </>
                ) : (
                  <p className="text-slate-500 italic">
                    {conversationId
                      ? "Writing sales prep…"
                      : "Start or open a conversation to generate prep."}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </details>
    </div>
  )
}
