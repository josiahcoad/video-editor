import { useCallback, useEffect, useRef, useState } from "react"
import {
  Navigate,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom"
import { Layout } from "@/components/Layout"
import { ContextBar } from "@/components/ContextBar"
import { TestBar } from "@/components/TestBar"
import { SessionView } from "@/components/SessionView"
import { ReviewOverlay } from "@/components/ReviewOverlay"
import { ContactsView } from "@/components/ContactsView"
import { ContactDetail } from "@/components/ContactDetail"
import { ConversationDetail } from "@/components/ConversationDetail"
import { HomeView } from "@/components/HomeView"
import {
  fetchContacts,
  fetchContactsWithSummary,
  fetchContactDetail,
  fetchConversation,
} from "@/lib/api"
import type {
  ContactDetail as ContactDetailType,
  ContactOption,
  ContactSummary,
  ConversationDetail as ConversationDetailType,
  Review,
} from "@/types/api"
import { useCoachWs } from "@/hooks/useCoachWs"

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  const [contactId, setContactId] = useState("")
  const [context, setContext] = useState("")
  const [contacts, setContacts] = useState<ContactOption[]>([])
  const [contactsSummary, setContactsSummary] = useState<ContactSummary[]>([])
  const [contactsLoading, setContactsLoading] = useState(false)
  const [contactsError, setContactsError] = useState<string | null>(null)
  const [reviewOverlayOpen, setReviewOverlayOpen] = useState(false)
  const [reviewOverlayLoading, setReviewOverlayLoading] = useState(false)
  const [review, setReview] = useState<Review | null>(null)
  const [autoCoach, setAutoCoach] = useState(true)
  const [critiqueMode, setCritiqueMode] = useState(false)
  const [running, setRunning] = useState(false)
  const [connecting, setConnecting] = useState(false)
  const [pausedConversationId, setPausedConversationId] = useState<string | null>(
    null
  )

  const {
    state: wsState,
    setState: setWsState,
    setStatus,
    startTimer,
    stopTimer,
    resetPanels,
    handleMessage,
    wsRef,
    defaultState,
  } = useCoachWs()

  const streamRef = useRef<MediaStream | null>(null)
  const audioCtxRef = useRef<AudioContext | null>(null)

  const stopLive = useCallback(
    (opts?: { skipEndMessage?: boolean }) => {
      setRunning(false)
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close()
        audioCtxRef.current = null
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
      stopTimer()
      setWsState((s) => ({ ...s, showInterim: false }))
      if (opts?.skipEndMessage) {
        setStatus("Paused", "gray")
        return
      }
      setStatus("Session ended — generating review…", "yellow")
      if (wsState.sessionId && wsState.tips.length > 0) {
        setReviewOverlayOpen(true)
        setReviewOverlayLoading(true)
        pollForReview(wsState.sessionId)
      }
    },
    [stopTimer, setStatus, setWsState, wsState.sessionId, wsState.tips.length]
  )

  async function pollForReview(convId: string) {
    for (let i = 0; i < 30; i++) {
      await new Promise((r) => setTimeout(r, 2000))
      try {
        const data = await fetchConversation(Number(convId))
        if (data.review) {
          setReview(data.review)
          setReviewOverlayLoading(false)
          setStatus("Review ready", "green")
          return
        }
      } catch {
        // keep polling
      }
    }
    setReviewOverlayLoading(false)
    setReview(null)
  }

  const loadContacts = useCallback(async () => {
    try {
      const list = await fetchContacts()
      setContacts(list)
    } catch (e) {
      console.error(e)
    }
  }, [])

  const loadContactsSummary = useCallback(async () => {
    setContactsLoading(true)
    setContactsError(null)
    try {
      const list = await fetchContactsWithSummary()
      setContactsSummary(list)
    } catch (e) {
      setContactsError(e instanceof Error ? e.message : "Failed to load")
    } finally {
      setContactsLoading(false)
    }
  }, [])

  useEffect(() => {
    loadContacts()
  }, [loadContacts])

  useEffect(() => {
    if (location.pathname.startsWith("/contacts")) {
      loadContactsSummary()
    }
  }, [location.pathname, loadContactsSummary])

  useEffect(() => {
    if (location.pathname === "/live") {
      setWsState((s) => ({ ...s, ...defaultState, coachMeVisible: !autoCoach }))
    }
    if (location.pathname === "/test") {
      setWsState((s) => ({ ...s, showInterim: false }))
    }
  }, [location.pathname, autoCoach, defaultState, setWsState])

  const connectLive = useCallback(
    async (resumeId?: number) => {
      setConnecting(true)
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: {
            sampleRate: 16000,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
          },
        })
        streamRef.current = stream
        const audioCtx = new AudioContext({ sampleRate: 16000 })
        audioCtxRef.current = audioCtx
        const src = audioCtx.createMediaStreamSource(stream)
        const proc = audioCtx.createScriptProcessor(2048, 1, 1)
        proc.onaudioprocess = (e) => {
          if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN)
            return
          const f = e.inputBuffer.getChannelData(0)
          const i = new Int16Array(f.length)
          for (let x = 0; x < f.length; x++) {
            const s = Math.max(-1, Math.min(1, f[x]))
            i[x] = s < 0 ? s * 0x8000 : s * 0x7fff
          }
          wsRef.current.send(i.buffer)
        }
        src.connect(proc)
        proc.connect(audioCtx.destination)

        const proto = window.location.protocol === "https:" ? "wss:" : "ws:"
        const ws = new WebSocket(`${proto}//${window.location.host}/ws/coach`)
        ws.binaryType = "arraybuffer"
        wsRef.current = ws

        ws.onopen = () => {
          ws.send(
            JSON.stringify({
              auto_coach: autoCoach,
              critique_mode: critiqueMode,
              contact_id: contactId || null,
              ...(resumeId != null && { conversation_id: resumeId }),
            })
          )
          setConnecting(false)
          setRunning(true)
          if (!resumeId) resetPanels()
          setWsState((s) => ({
            ...s,
            showInterim: true,
            showScoreBar: true,
            status: "Listening…",
            statusColor: "green",
          }))
          setStatus("Listening…", "green")
          startTimer()
        }
        ws.onmessage = (e) => {
          try {
            const msg = JSON.parse(e.data as string)
            handleMessage(
              msg,
              (r) => {
                setReview(r)
                setReviewOverlayOpen(true)
                setReviewOverlayLoading(false)
              },
              (convId) => {
                setPausedConversationId(String(convId))
                stopLive({ skipEndMessage: true })
              },
              () => stopLive()
            )
          } catch {
            // ignore
          }
        }
        ws.onclose = () => {
          if (running) stopLive()
        }
        ws.onerror = () => setStatus("Connection error", "red")
      } catch (err) {
        console.error(err)
        setStatus(
          err instanceof Error && err.name === "NotAllowedError"
            ? "Mic access denied"
            : err instanceof Error
              ? err.message
              : "Failed to connect",
          "red"
        )
      }
      setConnecting(false)
    },
    [
      context,
      contactId,
      autoCoach,
      critiqueMode,
      running,
      resetPanels,
      setWsState,
      setStatus,
      startTimer,
      handleMessage,
      wsRef,
      stopLive,
    ]
  )

  const connectTestWs = useCallback(
    (initPayload: object) => {
      setStatus("Connecting…", "yellow")
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:"
      const ws = new WebSocket(`${proto}//${window.location.host}/ws/test`)
      wsRef.current = ws
      ws.onopen = () => {
        ws.send(JSON.stringify(initPayload))
        setRunning(true)
        resetPanels()
        setStatus("Test mode", "green")
        setWsState((s) => ({
          ...s,
          showScoreBar: true,
          showTestReviewBtn: true,
        }))
      }
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data as string)
          handleMessage(
            msg,
            (r) => {
              setReview(r)
              setReviewOverlayOpen(true)
              setReviewOverlayLoading(false)
            },
            undefined,
            undefined
          )
        } catch {
          // ignore
        }
      }
      ws.onclose = () => {
        setRunning(false)
        setStatus("Disconnected", "gray")
      }
    },
    [resetPanels, setStatus, setWsState, handleMessage, wsRef]
  )

  const connectTest = useCallback(
    (conversationId: number) => {
      connectTestWs({
        conversation_id: conversationId,
        critique_mode: critiqueMode,
        contact_id: contactId || null,
      })
    },
    [contactId, critiqueMode, connectTestWs]
  )

  const connectTestFresh = useCallback(() => {
    connectTestWs({
      conversation: "",
      critique_mode: critiqueMode,
      contact_id: contactId || null,
    })
  }, [contactId, critiqueMode, connectTestWs])

  const sendWs = useCallback(
    (payload: object) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(payload))
      }
    },
    [wsRef]
  )

  const onAutoCoachChange = useCallback(
    (v: boolean) => {
      setAutoCoach(v)
      setWsState((s) => ({ ...s, coachMeVisible: !v }))
      sendWs({ type: "set_auto_coach", value: v })
    },
    [sendWs, setWsState]
  )

  const onCritiqueChange = useCallback(
    (v: boolean) => {
      setCritiqueMode(v)
      sendWs({ type: "set_critique_mode", value: v })
    },
    [sendWs]
  )

  const onGenerateReview = useCallback(() => {
    sendWs({ type: "generate_review" })
    setReviewOverlayOpen(true)
    setReviewOverlayLoading(true)
  }, [sendWs])

  const sessionButton =
    location.pathname === "/live" ? (
      <div className="flex items-center gap-2">
        {running ? (
          <>
            <button
              type="button"
              onClick={() => sendWs({ type: "pause" })}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold transition active:scale-95 bg-slate-600 hover:bg-slate-500"
            >
              Pause
            </button>
            <button
              type="button"
              onClick={() => {
                setPausedConversationId(null)
                stopLive()
              }}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold transition active:scale-95 bg-red-600 hover:bg-red-500"
            >
              End Conversation
            </button>
          </>
        ) : (
          <>
            {pausedConversationId && (
              <button
                type="button"
                disabled={connecting}
                onClick={() => connectLive(Number(pausedConversationId))}
                className="px-4 py-1.5 rounded-lg text-sm font-semibold transition active:scale-95 bg-blue-600 hover:bg-blue-500 disabled:opacity-50"
              >
                Resume Conversation
              </button>
            )}
            <button
              type="button"
              disabled={connecting}
              onClick={() => {
                setPausedConversationId(null)
                connectLive()
              }}
              className="px-4 py-1.5 rounded-lg text-sm font-semibold transition active:scale-95 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50"
            >
              {connecting ? "Connecting…" : "Start Conversation"}
            </button>
          </>
        )}
      </div>
    ) : null

  const handleSelectContact = useCallback(
    (id: number) => navigate(`/contacts/${id}`),
    [navigate]
  )

  const handleSelectConversation = useCallback(
    (convId: number, contactId: number) =>
      navigate(`/contacts/${contactId}/conversations/${convId}`),
    [navigate]
  )

  return (
    <div className="bg-slate-950 text-slate-100 h-screen flex flex-col overflow-hidden">
      <Layout
        statusText={wsState.status}
        statusColor={wsState.statusColor}
        timerSeconds={wsState.timerSeconds}
        timerVisible={wsState.timerVisible}
        sessionButton={sessionButton}
      />

      <Routes>
        <Route path="/" element={<Navigate to="/home" replace />} />
        <Route path="/home" element={<HomeView />} />
        <Route
          path="/live"
          element={
            <>
              <ContextBar
                contacts={contacts}
                contactId={contactId}
                onContactIdChange={setContactId}
                context={context}
                onContextChange={setContext}
                disabled={running}
                onContactsReload={loadContacts}
              />
              <SessionView
                turns={wsState.turns}
                tips={wsState.tips}
                objections={wsState.objections}
                interimText={wsState.interimText}
                showInterim={wsState.showInterim}
                showScoreBar={wsState.showScoreBar}
                closeScore={wsState.closeScore}
                scoreBarColor={wsState.scoreBarColor}
                stepsToClose={wsState.stepsToClose}
                autoCoach={autoCoach}
                critiqueMode={critiqueMode}
                onAutoCoachChange={onAutoCoachChange}
                onCritiqueChange={onCritiqueChange}
                coachingLoading={wsState.coachingLoading}
                onCoachMe={() => sendWs({ type: "coach_me" })}
                onAskQuestion={(q) => sendWs({ type: "coach_me", question: q })}
              />
            </>
          }
        />
        <Route
          path="/test"
          element={
            <>
              <ContextBar
                contacts={contacts}
                contactId={contactId}
                onContactIdChange={setContactId}
                disabled={running}
                onContactsReload={loadContacts}
              />
              <TestBar
                onSelectConversation={connectTest}
                onStartFresh={connectTestFresh}
                onGenerateReview={onGenerateReview}
                showReviewBtn={wsState.showTestReviewBtn}
              />
              <SessionView
                turns={wsState.turns}
                tips={wsState.tips}
                objections={wsState.objections}
                interimText={wsState.interimText}
                showInterim={wsState.showInterim}
                showScoreBar={wsState.showScoreBar}
                closeScore={wsState.closeScore}
                scoreBarColor={wsState.scoreBarColor}
                stepsToClose={wsState.stepsToClose}
                autoCoach={autoCoach}
                critiqueMode={critiqueMode}
                onAutoCoachChange={onAutoCoachChange}
                onCritiqueChange={onCritiqueChange}
                coachingLoading={wsState.coachingLoading}
                onCoachMe={() => sendWs({ type: "coach_me" })}
                onAskQuestion={(q) => sendWs({ type: "coach_me", question: q })}
                onCoachUpToTurn={(turn) =>
                  sendWs({ type: "coach_me", up_to_turn: turn })
                }
                onSubmitTurn={
                  running
                    ? (text) => sendWs({ type: "add_turn", text })
                    : undefined
                }
              />
            </>
          }
        />
        <Route
          path="/contacts"
          element={
            <ContactsView
              contacts={contactsSummary}
              loading={contactsLoading}
              error={contactsError}
              onSelectContact={handleSelectContact}
              onContactsReload={loadContactsSummary}
            />
          }
        />
        <Route
          path="/contacts/:id"
          element={
            <ContactDetailRoute
              onSelectConversation={handleSelectConversation}
              onStartConversation={(contactId) => {
                setContactId(String(contactId))
                navigate("/live")
              }}
            />
          }
        />
        <Route
          path="/contacts/:contactId/conversations/:convId"
          element={<ConversationDetailRoute />}
        />
      </Routes>

      <ReviewOverlay
        open={reviewOverlayOpen}
        onClose={() => {
          setReviewOverlayOpen(false)
          setReview(null)
        }}
        review={review}
        loading={reviewOverlayLoading}
      />
    </div>
  )
}

function ContactDetailRoute({
  onSelectConversation,
  onStartConversation,
}: {
  onSelectConversation: (convId: number, contactId: number) => void
  onStartConversation?: (contactId: number) => void
}) {
  const { id } = useParams<"id">()
  const navigate = useNavigate()
  const [contact, setContact] = useState<ContactDetailType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!id) return
    let cancelled = false
    fetchContactDetail(Number(id))
      .then((c) => {
        if (!cancelled) setContact(c)
      })
      .catch(() => {
        if (!cancelled) setContact(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [id])

  if (loading) {
    return (
      <main className="flex-1 min-h-0 flex items-center justify-center">
        <p className="text-slate-500">Loading contact…</p>
      </main>
    )
  }
  if (!contact) {
    return (
      <main className="flex-1 min-h-0 flex items-center justify-center gap-3">
        <p className="text-red-400">Contact not found</p>
        <button
          type="button"
          onClick={() => navigate("/contacts")}
          className="text-sm text-slate-400 hover:text-white"
        >
          ← Back to contacts
        </button>
      </main>
    )
  }

  return (
    <ContactDetail
      contact={contact}
      onBack={() => navigate("/contacts")}
      onSelectConversation={onSelectConversation}
      onStatusChange={() =>
        fetchContactDetail(contact.id).then(setContact)
      }
      onStartConversation={onStartConversation}
    />
  )
}

function ConversationDetailRoute() {
  const { contactId, convId } = useParams<"contactId" | "convId">()
  const navigate = useNavigate()
  const [conversation, setConversation] =
    useState<ConversationDetailType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    if (!convId) return
    let cancelled = false
    fetchConversation(Number(convId))
      .then((c) => {
        if (!cancelled) setConversation(c)
      })
      .catch(() => {
        if (!cancelled) setConversation(null)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [convId])

  if (loading) {
    return (
      <main className="flex-1 min-h-0 flex items-center justify-center">
        <p className="text-slate-500">Loading conversation…</p>
      </main>
    )
  }
  if (!conversation) {
    return (
      <main className="flex-1 min-h-0 flex items-center justify-center gap-3">
        <p className="text-red-400">Conversation not found</p>
        <button
          type="button"
          onClick={() =>
            navigate(contactId ? `/contacts/${contactId}` : "/contacts")
          }
          className="text-sm text-slate-400 hover:text-white"
        >
          ← Back
        </button>
      </main>
    )
  }

  return (
    <ConversationDetail
      conversation={conversation}
      onBack={() =>
        navigate(contactId ? `/contacts/${contactId}` : "/contacts")
      }
      onReviewGenerated={() =>
        fetchConversation(Number(convId)).then(setConversation)
      }
    />
  )
}
