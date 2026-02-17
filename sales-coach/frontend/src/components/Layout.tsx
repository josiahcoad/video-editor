import { NavLink } from "react-router-dom"
import { cn } from "@/lib/utils"

interface LayoutProps {
  statusText: string
  statusColor: "gray" | "green" | "yellow" | "red"
  timerSeconds: number
  timerVisible: boolean
  sessionButton: React.ReactNode
}

const statusDotClass: Record<LayoutProps["statusColor"], string> = {
  gray: "bg-slate-600",
  green: "bg-emerald-500 animate-pulse-dot",
  yellow: "bg-amber-500",
  red: "bg-red-500",
}

const tabConfig = [
  { path: "/home", label: "Home", end: true },
  { path: "/live", label: "Live", end: true },
  { path: "/test", label: "Test", end: true },
  { path: "/contacts", label: "Contacts", end: false },
] as const

export function Layout({
  statusText,
  statusColor,
  timerSeconds,
  timerVisible,
  sessionButton,
}: LayoutProps) {
  const pad = (n: number) => String(n).padStart(2, "0")
  const timerStr = `${pad(Math.floor(timerSeconds / 60))}:${pad(timerSeconds % 60)}`

  return (
    <header className="shrink-0 bg-slate-900/80 backdrop-blur border-b border-slate-800 px-6 py-3">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-2xl">🎯</span>
          <div>
            <h1 className="text-lg font-bold leading-tight">Sales Coach</h1>
            <p className="text-xs text-slate-500">
              Deepgram Flux · Claude Haiku 4.5
            </p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex rounded-lg overflow-hidden border border-slate-700 text-xs">
            {tabConfig.map(({ path, label, end }) => (
              <NavLink
                key={path}
                to={path}
                end={end}
                className={({ isActive }) =>
                  cn(
                    "px-3 py-1.5 font-medium capitalize",
                    isActive
                      ? "bg-slate-700 text-white"
                      : "bg-slate-800 text-slate-400 hover:text-white"
                  )
                }
              >
                {label}
              </NavLink>
            ))}
          </div>
          <div className="flex items-center gap-2 text-sm text-slate-500">
            <span
              className={cn("w-2 h-2 rounded-full", statusDotClass[statusColor])}
            />
            <span>{statusText}</span>
          </div>
          {timerVisible && (
            <span className="font-mono text-sm text-slate-600">{timerStr}</span>
          )}
          {sessionButton}
        </div>
      </div>
    </header>
  )
}
