import { useState } from "react"
import type { ContactOption } from "@/types/api"
import { createContact, createConversation } from "@/lib/api"

const STATUS_OPTIONS = [
  "prospect",
  "interested",
  "demoing",
  "reviewing_contract",
] as const

interface LiveStartViewProps {
  contacts: ContactOption[]
  onContactsReload: () => void
  /** Called with (conversationId, contactId) so parent can set contact and navigate to /live/:id */
  onStart: (conversationId: number, contactId: number) => void
}

export function LiveStartView({
  contacts,
  onContactsReload,
  onStart,
}: LiveStartViewProps) {
  const [selectedContactId, setSelectedContactId] = useState<string>("")
  const [creating, setCreating] = useState(false)
  const [showNew, setShowNew] = useState(false)
  const [newName, setNewName] = useState("")
  const [newCompany, setNewCompany] = useState("")
  const [newStatus, setNewStatus] = useState<string>("prospect")
  const [saving, setSaving] = useState(false)

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
      setSelectedContactId(String(c.id))
      setShowNew(false)
      setNewName("")
      setNewCompany("")
    } catch (e) {
      console.error(e)
    } finally {
      setSaving(false)
    }
  }

  async function handleStartConversation() {
    const contactId = selectedContactId ? Number(selectedContactId) : undefined
    if (contactId != null && Number.isNaN(contactId)) return
    setCreating(true)
    try {
      const { id } = await createConversation(contactId)
      onStart(id, contactId ?? 0)
    } catch (e) {
      console.error(e instanceof Error ? e.message : "Failed to create conversation")
    } finally {
      setCreating(false)
    }
  }

  const canStart = selectedContactId !== ""

  return (
    <main className="flex-1 min-h-0 max-w-xl w-full mx-auto p-6 flex flex-col justify-center">
      <div className="bg-slate-900 rounded-2xl border border-slate-800 p-6 space-y-6">
        <h2 className="text-lg font-bold text-white">
          Start a call
        </h2>
        <p className="text-sm text-slate-400">
          Choose a contact to create a new conversation. You’ll be taken to the
          call screen where you can start the session.
        </p>
        <div className="space-y-3">
          <label className="block text-xs font-medium text-slate-500 uppercase tracking-wide">
            Contact
          </label>
          <div className="flex flex-wrap gap-2">
            <select
              value={selectedContactId}
              onChange={(e) => setSelectedContactId(e.target.value)}
              className="flex-1 min-w-[200px] bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 focus:outline-none focus:border-emerald-600"
            >
              <option value="">Select a contact…</option>
              {contacts.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.company ? `${c.name} (${c.company})` : c.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              onClick={() => setShowNew(true)}
              className="px-3 py-2 text-sm rounded-lg bg-slate-700 hover:bg-slate-600 font-medium"
            >
              + New contact
            </button>
          </div>
          {showNew && (
            <div className="rounded-xl border border-slate-700 bg-slate-800/60 p-4 space-y-3">
              <input
                type="text"
                placeholder="Name *"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-600"
              />
              <input
                type="text"
                placeholder="Company"
                value={newCompany}
                onChange={(e) => setNewCompany(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-emerald-600"
              />
              <select
                value={newStatus}
                onChange={(e) => setNewStatus(e.target.value)}
                className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-[11px] text-slate-300 focus:outline-none"
              >
                {STATUS_OPTIONS.map((s) => (
                  <option key={s} value={s}>
                    {s === "reviewing_contract" ? "Reviewing Contract" : s}
                  </option>
                ))}
              </select>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={handleSaveNew}
                  disabled={saving || !newName.trim()}
                  className="px-3 py-1.5 text-sm rounded-lg bg-emerald-700 hover:bg-emerald-600 font-medium disabled:opacity-50"
                >
                  {saving ? "Saving…" : "Save"}
                </button>
                <button
                  type="button"
                  onClick={() => setShowNew(false)}
                  className="px-3 py-1.5 text-sm text-slate-500 hover:text-slate-300"
                >
                  Cancel
                </button>
              </div>
            </div>
          )}
        </div>
        <button
          type="button"
          onClick={handleStartConversation}
          disabled={!canStart || creating}
          className="w-full py-3 rounded-xl bg-emerald-600 hover:bg-emerald-500 font-semibold text-white disabled:opacity-50 disabled:cursor-not-allowed transition"
        >
          {creating ? "Creating…" : "Start conversation"}
        </button>
      </div>
    </main>
  )
}
