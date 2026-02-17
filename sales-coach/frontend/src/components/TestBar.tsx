import { useEffect, useState } from "react"
import { fetchConversations } from "@/lib/api"

export interface ConversationOption {
  id: number
  started_at: string | null
  mode: string
  contact_name?: string
  contact_company?: string | null
}

interface TestBarProps {
  onSelectConversation: (conversationId: number) => void
  onStartFresh: () => void
  onGenerateReview: () => void
  showReviewBtn: boolean
}

export function TestBar({
  onSelectConversation,
  onStartFresh,
  onGenerateReview,
  showReviewBtn,
}: TestBarProps) {
  const [conversations, setConversations] = useState<ConversationOption[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedId, setSelectedId] = useState<string>("")

  useEffect(() => {
    let cancelled = false
    async function load() {
      setLoading(true)
      try {
        const list = await fetchConversations()
        if (!cancelled) setConversations(list as ConversationOption[])
      } catch {
        if (!cancelled) setConversations([])
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    load()
    return () => {
      cancelled = true
    }
  }, [])

  function handleLoad() {
    const id = selectedId ? Number(selectedId) : 0
    if (!id) return
    onSelectConversation(id)
  }

  return (
    <div className="shrink-0 max-w-7xl w-full mx-auto px-4 pt-3 pb-0">
      <div className="bg-slate-900 rounded-xl border border-slate-800 px-4 py-3">
        <div className="flex items-center gap-3 flex-wrap">
          <label className="text-sm font-semibold text-slate-400">
            📞 Conversation to test
          </label>
          <select
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            disabled={loading}
            className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-sm text-slate-200 focus:outline-none focus:border-emerald-600 disabled:opacity-50 min-w-[200px]"
          >
            <option value="">Select a conversation…</option>
            {conversations.map((c) => {
              const date = c.started_at
                ? new Date(c.started_at).toLocaleDateString("en-US", {
                    month: "short",
                    day: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                  })
                : "—"
              const label = c.contact_name
                ? `${c.contact_name} · ${date}`
                : `#${c.id} · ${date}`
              return (
                <option key={c.id} value={c.id}>
                  {label}
                </option>
              )
            })}
          </select>
          <button
            type="button"
            onClick={handleLoad}
            disabled={loading || !selectedId}
            className="px-3 py-1.5 text-xs rounded-lg bg-blue-600 hover:bg-blue-500 font-medium disabled:opacity-50 disabled:pointer-events-none"
          >
            Load & Connect
          </button>
          <span className="text-xs text-slate-600">or</span>
          <button
            type="button"
            onClick={onStartFresh}
            className="px-3 py-1.5 text-xs rounded-lg bg-slate-700 hover:bg-slate-600 font-medium"
          >
            Start Fresh
          </button>
          {showReviewBtn && (
            <button
              type="button"
              onClick={onGenerateReview}
              className="px-3 py-1.5 text-xs rounded-lg bg-purple-600 hover:bg-purple-500 font-medium"
            >
              Generate Review
            </button>
          )}
        </div>
        {loading && (
          <p className="text-xs text-slate-500 mt-2">Loading conversations…</p>
        )}
      </div>
    </div>
  )
}
