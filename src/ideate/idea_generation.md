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

Optional: pass `--strategy projects/<ClientName>/strategy.md` or rely on the default (strategy.md next to profile.md when present). Ideate uses both profile and strategy for topic generation.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--profile` / `-p` | (required) | Path to brand profile (`.md` or `.json`) |
| `--output` / `-o` | (none) | Path to save JSON results |
| `--country` / `-c` | `us` | Country code for AnswerThePublic |
| `--max-topics` / `-n` | `20` | Max topics to return |

### What happens under the hood

1. **Load profile** — reads the brand definition (markdown or JSON)
2. **Propose search terms** — LLM generates 10 seed keywords (1-2 words each) based on the profile
3. **Query AnswerThePublic** — each keyword is sent to ATP, returning real search demand data
4. **Extract topics** — raw ATP responses are parsed into unique topic strings (expect 500-7000+ depending on keyword quality)
5. **Filter & rank** — LLM scores each topic against the brand profile on three axes:
   - **Relevance** (1-10): alignment with messaging pillars and brand
   - **Excitement** (1-10): how likely the client would enjoy talking about this
   - **Buyer intent** (1-10): how close the searcher is to the client's ICP
6. **Output** — top N topics with scores, rationale, and suggested angles

### Troubleshooting

- **"RAPID_API_KEY not set"** — the key isn't in `.env` or the variable name is wrong. Must be exactly `RAPID_API_KEY`.
- **0 topics extracted from ATP** — all ATP queries failed. Check your RapidAPI key and quota. The script will fall back to using the LLM's search terms directly, but results will be weaker.
- **Poor topic quality** — check the proposed search terms in the output. If they're 3+ words, the ATP results will be thin. The prompt constrains to 1-2 words, but if the LLM drifts, re-run.

---

## Writing the ideas markdown (`ideas.md`)

The script outputs raw JSON. Convert it into a human-readable markdown file for client review:

```
projects/<ClientName>/ideas.md
```

### Structure

The JSON output contains all fields needed for a 1-1 export to markdown. Follow this format (see existing client `ideas.md` files for examples):

```markdown
# Recording Session Ideas — <Client Name>

**Based on:** <what informed the ideas — profile, analysis, ATP data>
**Format:** <recording format — e.g., sub-35s direct-to-camera>
**Next step:** <what happens next — client picks 5-7>

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
| `hook` | Hook (3s open) | One simple sentence/fragment; no "When/If X, Y" compound |
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
