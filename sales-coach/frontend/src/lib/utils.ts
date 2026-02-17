import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Strip markdown code fences from text. Handles trailing content after closing fence. */
export function stripCodeFences(text: string): string {
  if (!text || typeof text !== "string") return text
  const t = text.trim()
  const m = t.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/)
  if (m) return m[1].trim()
  if (t.startsWith("```")) {
    return t.replace(/^```(?:json)?\s*/i, "").replace(/\s*```\s*$/, "").trim()
  }
  return t
}

/** Extract displayable advice from raw LLM response (JSON with "advice" key). */
export function extractAdviceDisplay(text: string): string {
  if (!text || typeof text !== "string") return text
  const stripped = stripCodeFences(text).trim()
  if (!stripped.startsWith("{")) return stripped
  let end = stripped.lastIndexOf("}")
  while (end >= 0) {
    try {
      const obj = JSON.parse(stripped.slice(0, end + 1)) as { advice?: string }
      if (typeof obj?.advice === "string" && obj.advice.trim()) return obj.advice.trim()
      break
    } catch {
      end = stripped.lastIndexOf("}", end - 1)
    }
  }
  return stripped
}
