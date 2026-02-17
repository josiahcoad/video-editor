# Stage 2: IDEATE — Idea Generation

Generate demand-backed topic ideas from a client's brand profile.

---

## Prerequisites

### 1. Brand profile (`profile.md`)

The idea generator requires a completed brand profile from the Profiler stage. This lives at:

```
projects/<ClientName>/profile.md
```

This file should contain (at minimum):
- ICP / audience description
- Messaging pillars
- Authority & expertise areas
- Brand voice & boundaries
- Content strategy inputs (what works, what bombs)
- Content production rules / constraints

See existing client profiles for examples (e.g. `projects/BillyTheCarKid/profile.md`, `projects/WyattZier/profile.md`).

**If the profile file doesn't exist yet**, create it first. Use the questionnaire and instructions in `src/profiler.md` as the starting point. Fill in what you can from public research, then flag `*NEEDS ANSWER*` for anything requiring a client interview. The Profiler stage produces two docs: **profile.md** (stable business profile) and **strategy.md** (content strategy, active doc).

### 2. Environment variables

The following must be set in `.env`:

| Variable | Purpose |
|----------|---------|
| `OPENROUTER_API_KEY` | LLM calls (search term generation + topic filtering) |
| `RAPID_API_KEY` | AnswerThePublic API via RapidAPI |

---

## Running the idea generator

From the workspace root:

```bash
dotenvx run -f .env -- uv run python -m src.ideate.ideate \
    --profile projects/<ClientName>/profile.md \
    --output projects/<ClientName>/ideas.json
```

If the client has a **strategy.md** (content strategy from the Profiler stage), ideate will use it automatically when it sits next to profile.md, or you can pass it explicitly with `--strategy`. The LLM receives both profile and strategy so topic ideas align with format, platforms, and positioning.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` / `-p` | (required) | Path to brand profile (`.md` or `.json`) |
| `--strategy` / `-s` | (none) | Path to content strategy (`.md`). If omitted and profile is `profile.md`, uses `strategy.md` in the same directory when present. |
| `--output` / `-o` | (none) | Path to save JSON results |
| `--country` / `-c` | `us` | Country code for AnswerThePublic |
| `--max-topics` / `-n` | `20` | Max topics to return |

### What happens under the hood

1. **Load profile (+ optional strategy)** — reads profile.md and, if present, strategy.md (same dir or `--strategy`). Combined context is used for all LLM steps.
2. **Propose search terms** — LLM generates 10 seed keywords (1-2 words each) based on the profile
3. **Query AnswerThePublic** — each keyword is sent to ATP, returning real search demand data
4. **Extract topics** — raw ATP responses are parsed into unique topic strings (expect 500-7000+ depending on keyword quality)
5. **Filter & rank** — LLM scores each topic against the brand profile on three axes:
   - **Relevance** (1-10): alignment with messaging pillars and brand
   - **Excitement** (1-10): how likely the client would enjoy talking about this
   - **Buyer intent** (1-10): how close the searcher is to the client's ICP
6. **Output** — writes both `ideas.json` and `ideas.md` (same directory as `--output`)

To regenerate only `ideas.md` from an existing `ideas.json` (e.g. after editing the template), run:

```bash
dotenvx run -f .env -- uv run python -m src.ideate.ideate --markdown-only -o projects/<ClientName>/ideas.json
```

### Troubleshooting

- **"RAPID_API_KEY not set"** — the key isn't in `.env` or the variable name is wrong. Must be exactly `RAPID_API_KEY`.
- **0 topics extracted from ATP** — all ATP queries failed. Check your RapidAPI key and quota. The script will fall back to using the LLM's search terms directly, but results will be weaker.
- **Poor topic quality** — check the proposed search terms in the output. If they're 3+ words, the ATP results will be thin. The prompt constrains to 1-2 words, but if the LLM drifts, re-run.

---

## The ideas markdown (`ideas.md`)

When you run the idea generator with `--output`, it writes both `ideas.json` and `ideas.md` in the same directory. The markdown is generated automatically with the summary table (checkbox, user search term, virality bars) and full idea blocks grouped by pillar.

To regenerate only `ideas.md` from existing JSON:  
`python -m src.ideate.ideate --markdown-only -o projects/<ClientName>/ideas.json`

### Structure

The JSON output contains all fields needed for a 1-1 export to markdown. Follow this format (see existing client `ideas.md` files for examples):

```markdown
# Recording Session Ideas — <Client Name>

**Based on:** <what informed the ideas — profile, analysis, ATP data>
**Format:** <recording format — e.g., sub-35s direct-to-camera>
**Next step:** <what happens next — client picks 5-7>

---

## Summary — Pick your batch

Both the checklist and the table are **sorted by virality** (highest first). The client can tick ideas in the checklist and use the table for prompt + hook at a glance.

### Checklist (virality-sorted, numbered)

- [ ] 1. &lt;AI-style user search query / user_search_term&gt;
- [ ] 2. ...
- [ ] N. ...

- **Format:** `- [ ] <number>. <user_search_term>` — no quotes. Number is 1..N in virality order.
- **User search term:** From JSON `user_search_term`. For discoverability in AI search, this should be a full natural-language question (e.g. "What are typical dealer fees added to new car purchases?" not "dealer fees").

### Table (virality-sorted)

| # | Prompt | Virality Score | Hook (3s open) |
|---|--------|----------------|----------------|
| 1 | &lt;prompt&gt; | X.X | &lt;hook&gt; |
| ... | ... | ... | ... |

- **#:** Row number in virality order (1 = highest).
- **Prompt:** The question to ask the founder to draw out the angle (from JSON `prompt`).
- **Virality Score:** `(relevance + excitement + buyer_intent) / 3` to one decimal.
- **Hook (3s open):** One simple sentence or fragment (~8–12 words), said first. No compound sentences (no "When X, Y" / "If X, Y") or the viewer tunes out (from JSON `hook`).

---

## Top 20 Ideas

### Pillar: "<Messaging Pillar Name>"

**1. "<Topic Title>"** — [Virality Score: X.X]
- ICP: <who this is for>
- Pillar: <which messaging pillar>
- Hook: "<one simple sentence, ~8–12 words, no compound clauses>"
- Prompt: "<question to ask the founder to naturally draw out the angle>"
- Angle: <the specific take, story, or argument>
- Rationale: <why this topic fits this specific founder>
- Source search keyword: "<original seed keyword>"
- User search term: "<actual long-tail query people are Googling>"

(repeat for each idea, grouped by pillar)

---

## Session Planning

**Recommended first session (pick 5–7):**

| # | Topic | Why |
|---|-------|-----|
| ... | ... | ... |

---

## Production Reminders

(client-specific recording constraints from the definition file)
```

### Field mapping (JSON → markdown)

All fields come directly from the JSON — no manual expansion needed:

| JSON field | Markdown field | Notes |
|------------|---------------|-------|
| `topic` | Topic title | — |
| `pillar` | Pillar | Group ideas by this |
| `icp` | ICP | — |
| `hook` | Hook (3s open) | One simple sentence/fragment (~8–12 words). No compound "When/If X, Y" — one idea only |
| `prompt` | Prompt | Question to ask the founder during recording |
| `angle` | Angle | The take / story / argument |
| `rationale` | Rationale | Why this fits the founder |
| `source_keyword` | Source search keyword | Which seed term produced this |
| `user_search_term` | User search term | The actual long-tail query |
| avg of 3 scores | Virality Score | `(relevance + excitement + intent) / 3` |

### Additional steps when writing the markdown

- **Group by messaging pillar**, not by score. The client thinks in themes, not numbers.
- **Recommend a first session batch** of 5-7 that covers different pillars and ICP segments for maximum variety.
- **Add production reminders** at the bottom — pulled from the definition file's content rules (duration limits, format constraints, etc.).

---

## Output files

After this stage, the client's project directory should contain:

```
projects/<ClientName>/
├── profile.md             # Brand profile (from Profiler stage; stable)
├── strategy.md            # Content strategy (from Profiler stage; active doc)
├── ideas.json             # Raw scored output from idea_generator
└── ideas.md               # Human-readable ideas for client review
```
