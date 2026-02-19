import type {
  ContactDetail,
  ContactOption,
  ContactSummary,
  ConversationDetail,
  ConversationSummary,
  HomeData,
  PerformanceReview,
  Review,
  TodoItem,
  TodoType,
} from "@/types/api"

const API = "/api"

export async function fetchContacts(): Promise<ContactOption[]> {
  const res = await fetch(`${API}/contacts`)
  if (!res.ok) throw new Error("Failed to fetch contacts")
  return res.json()
}

export async function fetchContactsWithSummary(): Promise<ContactSummary[]> {
  const res = await fetch(`${API}/contacts?summary=true`)
  if (!res.ok) throw new Error("Failed to fetch contacts")
  return res.json()
}

export async function fetchContactDetail(id: number): Promise<ContactDetail> {
  const res = await fetch(`${API}/contacts/${id}/detail`)
  if (!res.ok) throw new Error("Failed to fetch contact")
  const data = await res.json()
  if (data.error) throw new Error(data.error)
  return data
}

export async function askCoach(
  contactId: number,
  prompt: string
): Promise<{ response: string }> {
  const res = await fetch(`${API}/contacts/${contactId}/ask-coach`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error ?? "Failed to get coach response")
  }
  return res.json()
}

export async function createContact(body: {
  name: string
  company?: string | null
  phone?: string | null
  email?: string | null
  status: string
}): Promise<ContactOption & { id: number }> {
  const res = await fetch(`${API}/contacts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error("Failed to create contact")
  return res.json()
}

export async function updateContact(
  id: number,
  body: { status?: string; notes?: string; research?: string }
): Promise<void> {
  const res = await fetch(`${API}/contacts/${id}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error("Failed to update contact")
}

export async function deleteContact(id: number): Promise<void> {
  const res = await fetch(`${API}/contacts/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to delete contact")
}

export async function createConversation(contactId?: number): Promise<{ id: number }> {
  const res = await fetch(`${API}/conversations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(contactId != null ? { contact_id: contactId } : {}),
  })
  if (!res.ok) throw new Error("Failed to create conversation")
  return res.json()
}

export async function fetchConversations(
  contactId?: number
): Promise<ConversationSummary[]> {
  const url = contactId
    ? `${API}/conversations?contact_id=${contactId}`
    : `${API}/conversations`
  const res = await fetch(url)
  if (!res.ok) throw new Error("Failed to fetch conversations")
  return res.json()
}

export async function fetchConversation(id: number): Promise<ConversationDetail> {
  const res = await fetch(`${API}/conversations/${id}`)
  if (!res.ok) throw new Error("Failed to fetch conversation")
  return res.json()
}

export async function updateConversationPrepNotes(
  conversationId: number,
  prepNotes: string
): Promise<void> {
  const res = await fetch(`${API}/conversations/${conversationId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prep_notes: prepNotes }),
  })
  if (!res.ok) throw new Error("Failed to update conversation prep")
}

export async function deleteConversation(id: number): Promise<void> {
  const res = await fetch(`${API}/conversations/${id}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to delete conversation")
}

export async function postConversationReview(id: number): Promise<Review> {
  const res = await fetch(`${API}/conversations/${id}/review`, {
    method: "POST",
  })
  if (!res.ok) throw new Error("Failed to generate review")
  return res.json()
}

export async function fetchStatuses(): Promise<string[]> {
  const res = await fetch(`${API}/contacts/statuses`)
  if (!res.ok) return []
  const data = await res.json()
  return data.statuses ?? []
}

export async function fetchHome(sellerId: number = 1): Promise<HomeData> {
  const res = await fetch(`${API}/home?seller_id=${sellerId}`)
  if (!res.ok) throw new Error("Failed to fetch home data")
  return res.json()
}

export async function fetchQuickActions(
  sellerId: number = 1
): Promise<{ actions: string[] }> {
  const res = await fetch(`${API}/sellers/${sellerId}/quick-actions`)
  if (!res.ok) throw new Error("Failed to fetch quick actions")
  return res.json()
}

export async function fetchTodos(
  sellerId: number
): Promise<{ todos: TodoItem[] }> {
  const res = await fetch(`${API}/sellers/${sellerId}/todos`)
  if (!res.ok) throw new Error("Failed to fetch todos")
  return res.json()
}

export async function createTodo(
  sellerId: number,
  body: { type: TodoType; title: string; contact_id?: number | null }
): Promise<{ id: number }> {
  const res = await fetch(`${API}/sellers/${sellerId}/todos`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || "Failed to create todo")
  }
  return res.json()
}

export async function deleteTodo(todoId: number): Promise<void> {
  const res = await fetch(`${API}/todos/${todoId}`, { method: "DELETE" })
  if (!res.ok) throw new Error("Failed to delete todo")
}

export async function suggestTodos(
  sellerId: number
): Promise<{ created: number; todos: TodoItem[] }> {
  const res = await fetch(`${API}/sellers/${sellerId}/suggest-todos`, {
    method: "POST",
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || "Failed to suggest todos")
  }
  return res.json()
}

export async function fetchPrepare(contactId: number): Promise<{ content: string }> {
  const res = await fetch(`${API}/contacts/${contactId}/prepare`)
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || "Failed to generate preparation")
  }
  return res.json()
}

/** Returns the streaming response body; read with response.body.getReader() and TextDecoder. */
export async function fetchPrepareStream(
  contactId: number
): Promise<Response> {
  const res = await fetch(`${API}/contacts/${contactId}/prepare/stream`)
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || "Failed to start preparation stream")
  }
  return res
}

export async function fetchColdCallPrep(
  contactName: string,
  company: string | null,
  sellerId?: number | null
): Promise<{ content: string }> {
  const body: { contact_name: string; company: string | null; seller_id?: number } = {
    contact_name: contactName,
    company: company ?? null,
  }
  if (sellerId != null) body.seller_id = sellerId
  const res = await fetch(`${API}/cold-call-prep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new Error(data.error || "Failed to fetch cold call prep")
  }
  return res.json()
}

export async function generatePerformanceReview(
  sellerId: number = 1
): Promise<PerformanceReview> {
  const res = await fetch(`${API}/sellers/${sellerId}/performance-review`, {
    method: "POST",
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || "Failed to generate performance review")
  }
  return res.json()
}
