import { useCallback, useRef, useState } from "react"
import type { Hesitation, Review, WsMessage } from "@/types/api"

export interface TurnEntry {
  turn: number
  text: string
  confidence: number
  pending?: boolean
}

export interface TipEntry {
  id: string
  number: number
  turn: number
  advice: string
  closeScore?: number
  stepsToClose?: number
  isCustom?: boolean
  candidates?: Array<{
    one_liner?: string
    action: string
    trajectory_notes?: string
    close_probability?: number
  }>
}

export interface SessionState {
  status: string
  statusColor: "gray" | "green" | "yellow" | "red"
  timerVisible: boolean
  timerSeconds: number
  turns: TurnEntry[]
  tips: TipEntry[]
  objections: Hesitation[]
  interimText: string
  showInterim: boolean
  showScoreBar: boolean
  closeScore: number | null
  scoreBarColor: "emerald" | "amber" | "red"
  stepsToClose: number | null
  sessionId: string | null
  coachMeVisible: boolean
  showTestReviewBtn: boolean
  coachingLoading: boolean
}

const defaultState: SessionState = {
  status: "Ready",
  statusColor: "gray",
  timerVisible: false,
  timerSeconds: 0,
  turns: [],
  tips: [],
  objections: [],
  interimText: "",
  showInterim: false,
  showScoreBar: false,
  closeScore: null,
  scoreBarColor: "emerald",
  stepsToClose: null,
  sessionId: null,
  coachMeVisible: false,
  showTestReviewBtn: false,
  coachingLoading: false,
}

export function useCoachWs() {
  const [state, setState] = useState<SessionState>(defaultState)
  const wsRef = useRef<WebSocket | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const timerStartRef = useRef(0)
  const tipIdRef = useRef(0)

  const setStatus = useCallback(
    (text: string, color: SessionState["statusColor"]) => {
      setState((s) => ({ ...s, status: text, statusColor: color }))
    },
    []
  )

  const startTimer = useCallback(() => {
    timerStartRef.current = Date.now()
    setState((s) => ({ ...s, timerVisible: true, timerSeconds: 0 }))
    timerRef.current = setInterval(() => {
      const sec = Math.floor((Date.now() - timerStartRef.current) / 1000)
      setState((s) => ({ ...s, timerSeconds: sec }))
    }, 1000)
  }, [])

  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
    setState((s) => ({ ...s, timerVisible: false }))
  }, [])

  const resetPanels = useCallback(() => {
    tipIdRef.current = 0
    setState((s) => ({
      ...s,
      turns: [],
      tips: [],
      objections: [],
      interimText: "",
      closeScore: null,
      stepsToClose: null,
      sessionId: null,
      coachingLoading: false,
    }))
  }, [])

  const handleMessage = useCallback(
    (
      msg: WsMessage,
      onReview?: (review: Review) => void,
      onPaused?: (conversationId: number) => void,
      onAutoEnd?: () => void
    ) => {
      switch (msg.type) {
        case "session_id":
          setState((s) => ({ ...s, sessionId: msg.id }))
          break
        case "transcript_interim":
          setState((s) => ({
            ...s,
            interimText: msg.text,
            showInterim: true,
          }))
          break
        case "turn_end": {
          setState((s) => ({
            ...s,
            turns: [
              ...s.turns,
              {
                turn: msg.turn,
                text: msg.text,
                confidence: msg.confidence,
                pending: msg.pending,
              },
            ],
            interimText: "",
            showInterim: false,
          }))
          break
        }
        case "coaching_started":
          setState((s) => ({ ...s, coachingLoading: true }))
          break
        case "coaching": {
          const id = `tip-${++tipIdRef.current}`
          const tip: TipEntry = {
            id,
            number: msg.number,
            turn: msg.turn,
            advice: msg.advice,
            closeScore: msg.close_score,
            stepsToClose: msg.steps_to_close,
            isCustom: msg.is_custom,
            candidates: msg.candidates,
          }
          let scoreColor: SessionState["scoreBarColor"] = "emerald"
          if (msg.close_score != null && msg.close_score >= 0) {
            if (msg.close_score >= 70) scoreColor = "emerald"
            else if (msg.close_score >= 40) scoreColor = "amber"
            else scoreColor = "red"
          }
          setState((s) => ({
            ...s,
            tips: [tip, ...s.tips],
            coachingLoading: false,
            closeScore:
              msg.close_score != null && msg.close_score >= 0
                ? msg.close_score
                : s.closeScore,
            stepsToClose:
              msg.steps_to_close != null && msg.steps_to_close >= 0
                ? msg.steps_to_close
                : s.stepsToClose,
            scoreBarColor: scoreColor,
            showScoreBar: true,
            objections:
              "objections" in msg && Array.isArray(msg.objections)
                ? msg.objections
                : s.objections,
          }))
          break
        }
        case "review":
          onReview?.(msg.review)
          setStatus("Review ready", "green")
          break
        case "connected":
        case "status":
          setStatus(msg.message, "green")
          break
        case "paused":
          onPaused?.(msg.conversation_id)
          break
        case "auto_end":
          onAutoEnd?.()
          break
        case "error": {
          const errorTip: TipEntry = {
            id: `err-${Date.now()}`,
            number: 0,
            turn: 0,
            advice: msg.message,
            isCustom: false,
          }
          setState((s) => ({
            ...s,
            coachingLoading: false,
            tips: [{ ...errorTip, id: errorTip.id }, ...s.tips],
          }))
          break
        }
      }
    },
    [setStatus]
  )

  return {
    state,
    setState,
    setStatus,
    startTimer,
    stopTimer,
    resetPanels,
    handleMessage,
    wsRef,
    defaultState,
  }
}
