# Experimental Design: Anthony Reich TikTok Posting Optimization

**Date:** February 9, 2026
**Subject:** Anthony Reich (@texas_carguy), GM at Huffines Hyundai McKinney
**Platform:** TikTok
**Basis:** Observational analysis of 185 videos (see `analyze.md`)

---

## Background

Our observational analysis identified several factors that correlate with top-performing videos (duration, content archetype, question titles, text overlay frequency). However, two categories of factors remain inconclusive:

1. **Zero-variance factors** — posting time was always midnight (scheduled). We literally have no data on whether time-of-day matters for this account.
2. **Observational-only signals** — text overlay change frequency (1.4s vs 3.1s) was the strongest editing signal, but it's confounded with content quality. We don't know if it's causal.

This experiment is designed to gather maximum information from a small number of posts (~16) on the factors we couldn't resolve observationally.

---

## Factor Selection

### Why These Two Factors

| Factor | Rationale | Current Evidence |
|---|---|---|
| **Posting time** | All 185 posts were scheduled at midnight-1am. Complete blind spot. Trivially controllable via Marky scheduling. Zero cost to content quality. | None — zero variance in data |
| **Text overlay frequency** | Strongest editing signal found (top videos change text 2x faster). But only observational — could be confounded with content. Causal validation is high value. | Directional (1.4s vs 3.1s, observational) |

### Why NOT Other Factors

| Factor | Why excluded |
|---|---|
| **WPM / speaking pace** | Anthony speaks at ~206 WPM consistently. Asking him to artificially speed up or slow down produces unnatural content. You'd be testing "does talking weirdly hurt performance," not the actual variable. |
| **Duration** | Already have strong observational signal (42s vs 68s). And you can't control it independently — a 15s video and a 60s video are inherently different content pieces. |
| **Content theme** | Can't randomize. Assigning "talk about leadership today, car tips tomorrow" doesn't control for effort and authenticity differences. |
| **Question vs statement title** | Already have reasonable observational evidence (55% vs 20% in top vs bottom). Easy to test later if needed, but lower info-gain per post than the two chosen factors. |
| **Hashtags** | Low expected signal. Found nothing observationally. Could revisit in a future round if the first two factors resolve quickly. |
| **CTA presence** | Weak observational signal (45% vs 35%). Not worth the post budget. |

---

## Design: 2×2 Factorial

### Factors and Levels

**Factor A — Posting Time**

| Level | Time | Rationale |
|---|---|---|
| Morning | 7:00 AM CT | Catches commute / early scroll |
| Evening | 5:30 PM CT | Catches post-work / dinner scroll |

**Factor B — Text Overlay Style**

| Level | Description | Rationale |
|---|---|---|
| Rapid | 1-2 words per overlay, changing every ~1 second | Matches the pattern seen in top 20 observationally |
| Standard | Full short phrase per overlay, changing every ~3 seconds | Matches the pattern seen in bottom 20 observationally |

### Cell Assignment

| Cell | Posting Time | Text Style | Posts |
|---|---|---|---|
| A | Morning (7am) | Rapid (~1s) | 4 |
| B | Morning (7am) | Standard (~3s) | 4 |
| C | Evening (5:30pm) | Rapid (~1s) | 4 |
| D | Evening (5:30pm) | Standard (~3s) | 4 |
| **Total** | | | **16** |

### Controlled Variables (Hold Constant)

All 16 posts must follow these constraints to isolate the experimental factors:

- **Duration:** 15-30 seconds
- **Content:** Leadership topic, direct-to-camera, single point
- **Title:** Question format (e.g., "What's the #1 thing great leaders do differently?")
- **Thumbnail:** Text overlay present
- **Audio:** No background music
- **Format:** Single take, no b-roll, no interview, no screen recording
- **Editor:** Same person edits all 16 videos
- **Hashtags:** Same set of 3-5 hashtags on every post (removes hashtag variance)

---

## Randomized Schedule

Assuming 3 posts per week (Mon / Wed / Fri), the experiment runs ~5.5 weeks.

Cells are shuffled so that no factor is confounded with day-of-week or week number:

| Week | Monday | Wednesday | Friday |
|---|---|---|---|
| 1 | C (Eve/Rapid) | A (Morn/Rapid) | B (Morn/Std) |
| 2 | D (Eve/Std) | B (Morn/Std) | A (Morn/Rapid) |
| 3 | A (Morn/Rapid) | C (Eve/Rapid) | D (Eve/Std) |
| 4 | B (Morn/Std) | D (Eve/Std) | C (Eve/Rapid) |
| 5 | D (Eve/Std) | A (Morn/Rapid) | C (Eve/Rapid) |
| 6 | B (Morn/Std) | — | — |

### Balance Check

- Each cell appears on Monday at least once, Wednesday at least once, Friday at least once → no day-of-week confound
- Each cell appears in early weeks and late weeks → no time-trend confound
- Morning and Evening each get exactly 8 posts
- Rapid and Standard each get exactly 8 posts

---

## Measurement Protocol

### Primary Metric

**Views at 48 hours** — captures TikTok's initial algorithmic distribution push, before organic sharing and search take over.

### Secondary Metrics

| Metric | Captured at |
|---|---|
| Views | 48h, 7d |
| Likes | 48h, 7d |
| Comments | 48h, 7d |
| Shares | 48h, 7d |
| Watch time (if available via API) | 48h |

### Data Collection

- Pull stats via TikTok Content API at exactly 48h and 7d after each post
- Log any anomalies: TikTok outages, account events, unexpected virality, content that deviated from the controlled format
- Use **log(views)** for analysis — raw views are heavily right-skewed

### Analysis Plan

1. **Main effects** — compare Morning vs Evening (averaging across text styles) and Rapid vs Standard (averaging across times)
2. **Interaction** — does rapid text work better at a specific time of day?
3. **Effect size** — Cohen's d or ratio of geometric means (since we use log scale)
4. **Visualization** — box plots of log(views_48h) per cell

Use a 2-way ANOVA on log(views_48h) if assumptions are met, or Mann-Whitney U per factor if they're not (n=8 per level is marginal for parametric tests).

---

## Limitations and Honest Expectations

### What This CAN Tell Us

- **Directional evidence** — if morning outperforms evening in 6+ of 8 matched comparisons, that's informative even without formal significance
- **Whether the text overlay signal replicates** under controlled conditions
- **Whether there's an interaction** (e.g., rapid text + morning is the winning combo)
- **Order-of-magnitude effect size** — is posting time a 2x difference or a 5% difference?

### What This CANNOT Tell Us

- **Statistical significance is unlikely.** TikTok views have standard deviations of ~2-5x the mean. To detect a 50% lift at p<0.05 with 80% power, you'd need ~30 posts per cell (~120 total). This is a pilot.
- **We can't generalize** to other creators, niches, or platforms.
- **Confounds still exist** — TikTok's algorithm may change during the 6-week window. Follower growth shifts the baseline. One video going randomly viral inflates its cell's average.

### Decision Framework

| Outcome | Next Step |
|---|---|
| One factor shows 2x+ directional lift | Extend that factor for another 16 posts to confirm |
| Both factors show modest directional lift | Extend full experiment for another 16 posts |
| No directional signal in either factor | Call it — these factors don't matter enough for this creator. Refocus on content. |
| One cell dominates (interaction effect) | Adopt that combo as default, test a third factor in the next round |

---

## Instructions for Anthony / Editor

> **For the next 16 videos:**
>
> 1. Same format as your best stuff — direct-to-camera, one leadership point, under 30 seconds, question in the title.
> 2. We'll tell you when each video should be posted (either 7am or 5:30pm). Don't schedule them yourself.
> 3. For text overlays: some videos, break the text into very short chunks (1-2 words at a time, changing fast). Others, use normal-length phrases. We'll tell you which style for each video.
> 4. Use the same 3-5 hashtags on every post.
> 5. No music, no b-roll, no interviews. Just you, camera, one idea.
> 6. Everything else: business as usual.

---

## Timeline

| Milestone | Date (approx) |
|---|---|
| Experiment start | Week of Feb 17, 2026 |
| Midpoint check (8 posts) | Week of Mar 3, 2026 |
| All 16 posts published | Week of Mar 24, 2026 |
| Final 7-day metrics collected | Mar 31, 2026 |
| Analysis and decision | Apr 1, 2026 |

---

## Related Files

| File | Description |
|---|---|
| `analyze.md` | Full observational analysis (Phases 1-7) that informed this experiment |
| `anthony_reich_tiktok_analysis.csv` | Observational dataset (185 videos) |
| `editing_analysis_results.json` | Gemini video analysis results used to identify text overlay signal |
| `huffines_leaderboard.py` | Leaderboard script (separate from this analysis) |
