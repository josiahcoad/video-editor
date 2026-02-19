/** Contact list item (summary) */
export interface ContactSummary {
  id: number
  name: string
  company: string | null
  phone: string | null
  email: string | null
  seller_id?: number | null
  status: string
  next_step: string | null
  open_hesitation_count: number
  latest_close_score: number | null
  conversation_count: number
  /** ISO date string of most recent conversation */
  last_conversation_at: string | null
}

/** Contact for dropdown */
export interface ContactOption {
  id: number
  name: string
  company: string | null
  phone: string | null
  email: string | null
}

/** Contact detail (single contact with conversations) */
export interface ContactDetail extends ContactOption {
  seller_id?: number | null
  status: string
  next_step: string | null
  notes: string | null
  research?: string | null
  open_hesitations: Hesitation[]
  conversations: ConversationSummary[]
}

export interface Hesitation {
  text: string
  status: "open" | "resolved"
  rank?: "S" | "M" | "L"
  resolution_suggestion?: string
}

export interface ConversationSummary {
  id: number
  contact_id: number | null
  started_at: string | null
  mode: string
  next_step: string | null
  /** When set, this conversation's "next step" is linked to this todo. */
  next_step_todo_id: number | null
  /** 1–10 effectiveness from post-call review. */
  review_score: number | null
  final_close_score: number | null
  final_steps_to_close: number | null
  review_generated_at: string | null
  context?: string | null
  prep_notes?: string | null
}

export interface ConversationDetail extends ConversationSummary {
  contact_name: string
  contact_company: string | null
  transcript: TranscriptTurn[]
  coaching: CoachingTip[]
  hesitations: Hesitation[]
  review: Review | null
}

export interface TranscriptTurn {
  turn: number
  text: string
}

export interface CoachingTip {
  number: number
  turn: number
  advice: string
  /** Time from turn end to advice ready (ms). */
  latency_ms?: number | null
  close_score: number
  steps_to_close: number
  is_custom?: boolean
}

export interface Review {
  score: number
  went_well: string[]
  improve: string[]
  hesitations_summary?: {
    uncovered: string[]
    handled: string[]
    remaining: string[]
  }
  key_moment?: string
  follow_ups: string[]
  next_step?: string
  next_call_prep?: string
  suggested_status?: string
}

// ── Home / Sales Reps ────────────────────────────────────────────

export type TodoType = "email" | "call" | "other"

export interface TodoItem {
  id: number
  seller_id: number
  type: TodoType
  title: string
  contact_id: number | null
  contact_name: string | null
  created_at: string | null
}

export interface PipelineStage {
  status: string
  count: number
}

export interface PerformanceReview {
  overall_assessment: string
  strengths: string[]
  areas_for_improvement: string[]
  common_mistakes: string[]
  coaching_recommendations: string[]
  score: number
}

export interface CallsPerDay {
  date: string
  count: number
}

export interface HomeData {
  seller: {
    id: number
    first_name: string
    last_name: string
    company_info: string | null
  }
  pipeline: PipelineStage[]
  total_contacts: number
  total_conversations: number
  /** ISO started_at per conversation (backend UTC window); frontend groups by local date */
  conversation_start_times: string[]
  performance_review: PerformanceReview | null
  review_generated_at: string | null
}

/** WebSocket message types (incoming) */
export type WsMessage =
  | { type: "session_id"; id: string }
  | { type: "transcript_interim"; text: string }
  | { type: "turn_end"; turn: number; text: string; confidence: number; pending?: boolean }
  | { type: "coaching_started" }
  | { type: "paused"; conversation_id: number }
  | { type: "auto_end"; reason?: string }
  | {
      type: "coaching"
      number: number
      turn: number
      advice: string
      /** Time from turn end to advice ready (ms). */
      latency_ms?: number
      close_score?: number
      steps_to_close?: number
      is_custom?: boolean
      objections?: Hesitation[]
      candidates?: Array<{
        one_liner?: string
        action: string
        trajectory_notes?: string
        close_probability?: number
      }>
    }
  | {
      type: "scores"
      close_score: number
      steps_to_close: number
      objections: Hesitation[]
    }
  | { type: "review"; review: Review }
  | { type: "connected" | "status"; message: string }
  | { type: "error"; message: string }
