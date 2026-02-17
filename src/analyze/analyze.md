# TikTok Performance Analysis: Anthony Reich (@texas_carguy)

**Date:** February 9, 2026
**Analyst:** Marky AI + Jo
**Subject:** Anthony Reich, General Manager at Huffines Hyundai McKinney
**Platform:** TikTok
**Dataset:** 185 videos (Oct 2024 – Feb 2026)

---

## Table of Contents

1. [Objective](#objective)
2. [Methodology](#methodology)
3. [Phase 1: Data Collection](#phase-1-data-collection)
4. [Phase 2: Feature Importance (Quantitative)](#phase-2-feature-importance-quantitative)
5. [Phase 3: Enhanced Feature Analysis with Cross-Validation](#phase-3-enhanced-feature-analysis-with-cross-validation)
6. [Phase 4: Qualitative Top 20 vs Bottom 20 Comparison](#phase-4-qualitative-top-20-vs-bottom-20-comparison)
7. [Phase 5: Speech Cadence Analysis (Deepgram)](#phase-5-speech-cadence-analysis-deepgram)
8. [Phase 6: Editing Style Analysis (Gemini Video)](#phase-6-editing-style-analysis-gemini-video)
9. [Phase 7: Content Archetype Breakdown](#phase-7-content-archetype-breakdown)
10. [Findings Summary](#findings-summary)
11. [Actionable Recommendations](#actionable-recommendations)
12. [Strategic Product Insights for Marky](#strategic-product-insights-for-marky)
13. [Appendix: Methodology Details](#appendix-methodology-details)

---

## Objective

Determine what makes some TikTok videos outperform others for a car dealership GM, and produce actionable content guidelines that can be used to improve future video performance.

---

## Methodology

### Data Sources & Tools

| Source | Purpose |
|---|---|
| Marky production database (Supabase) | User, integration, and post metadata |
| TikTok Content API (`open.tiktokapis.com/v2`) | Video stats: views, likes, comments, shares, duration, thumbnails |
| Deepgram (nova-2) | Audio transcription with word-level timestamps |
| Google Gemini (2.0-flash, 2.5-flash) | Hook quality scoring, content classification, thumbnail analysis |
| scikit-learn (RF, GB, DT) | Feature importance and cross-validated modeling |

### Key Metric: Views Per Day (VPD)

Raw view counts are biased — older videos naturally accumulate more views. We normalized to `views_per_day = views / days_since_posted` to compare videos fairly across time periods.

### Pipeline

```
Supabase (user/integration lookup)
  → TikTok API (fetch all 185 videos + stats)
  → Deepgram (transcribe audio → word-level timestamps)
  → Gemini (thumbnail analysis, hook quality scoring 1-5, content classification)
  → scikit-learn (feature importance, cross-validation)
  → Manual qualitative analysis (top 20 vs bottom 20)
```

---

## Phase 1: Data Collection

Pulled 185 videos via TikTok Content API for the authenticated integration. For each video, collected:

- **Basic metrics:** views, likes, comments, shares, duration
- **Temporal:** posting date, day of week, hour of day
- **Content:** title, hashtags, @mentions, call-to-action presence
- **Audio:** Full transcript via Deepgram (URL-based transcription for Marky-stored videos, yt-dlp download + file transcription for older videos)
- **Hook:** First 3 seconds of spoken words (extracted from word timestamps)
- **Thumbnail:** Gemini multimodal analysis (text overlay, person presence, pointing gesture, car presence, setting)

Output: `anthony_reich_tiktok_analysis.csv` (185 rows, 15+ columns)

---

## Phase 2: Feature Importance (Quantitative)

### Round 1: Naive Feature Importance

Ran Decision Tree, Random Forest, and Gradient Boosting on raw features against raw `views` target.

**Initial findings (pre-validation):**

| Rank | Feature | Importance (RF) |
|---|---|---|
| 1 | duration_seconds | Highest across all models |
| 2 | day_of_week | 33% importance for likes (GB) |
| 3 | title_word_count | Correlated with performance |
| 4 | posting_hour | Moderate signal |

**Problem:** Gradient Boosting R² of 0.98-0.99 was suspiciously high — classic overfitting on 185 samples with no train/test split. Random Forest R² of 0.56-0.78 was more realistic but still inflated.

---

## Phase 3: Enhanced Feature Analysis with Cross-Validation

Added new features and applied proper validation:

### New Features Added

- `views_per_day` — normalized target (views / days since posted)
- `has_mention` — boolean: @mentions in title
- `days_since_last_post` — posting cadence
- `hook_quality_score` — Gemini-rated 1-5 (36 hooks scored)
- `days_since_first_video` — account momentum proxy
- `thumb_has_text_overlay`, `thumb_has_person`, `thumb_has_car`, `thumb_has_pointing` — extracted from Gemini thumbnail descriptions
- `title_has_emoji`, `title_has_question`, `hook_word_count`

### Results (5-fold Cross-Validation)

**All cross-validated R² values were negative** (-0.19 to -1.58).

This means the models perform *worse than predicting the mean*. With proper validation, these metadata features have **zero predictive power** for views/day, likes/day, or engagement rate.

### Key Findings

| Finding | Detail |
|---|---|
| `days_since_first_video` dominated (82%) | This is a confound — it captures audience growth over time, not content quality. Newer videos get more views because the account has more followers now. |
| Views/Day vs Duration (r = -0.079) | Essentially zero correlation. Duration is noise. |
| `title_word_count` (6%) | Weak second — possibly a proxy for "creator put more thought into it" |
| `hook_quality` (2%) | Shows up but sample too small (36 hooks) to be meaningful |

### Conclusion

**Metadata features (day, hour, duration, hashtags, emojis, etc.) do not predict which videos will go viral.** The signal is almost entirely in content quality and TikTok's algorithmic distribution — things that can't be captured from metadata alone.

---

## Phase 4: Qualitative Top 20 vs Bottom 20 Comparison

Since quantitative modeling failed, we pivoted to a qualitative comparison: rank all 185 videos by `views_per_day`, then compare the top 20 vs bottom 20 across multiple dimensions.

### Results

| Dimension | Top 20 | Bottom 20 | Signal? |
|---|---|---|---|
| **Avg duration** | 42s (median 32s) | 68s (median 60s) | **Strong** |
| **Question in title** | 11/20 (55%) | 4/20 (20%) | **Strong** |
| **Leadership content** | 12/20 (60%) | 5/20 (25%) | **Strong** |
| **Car-buying advice** | 0/20 (0%) | 4/20 (20%) | **Strong (inverse)** |
| **Text overlay on thumbnail** | 20/20 (100%) | 14/20 (70%) | **Moderate** |
| **CTA present** | 9/20 (45%) | 7/20 (35%) | Weak |
| **@mentions** | 5/20 (25%) | 5/20 (25%) | None |
| **Day of week** | Evenly distributed | Evenly distributed | None |
| **Posting hour** | Mostly midnight-1am | Mostly midnight-1am | None |

### What Top Performers Look Like

The winning template is clear: **short (under 35s), direct-to-camera, one clear leadership point, posed as a question, text overlay thumbnail.**

Examples of top performers:
- "Can you tell the difference between a manager and a leader?" (841 vpd, 20s)
- "How do you keep your team crystal clear on what matters most?" (703 vpd, 15s)
- "What's your secret to staying in the game long-term?" (518 vpd, 31s)

### What Bottom Performers Look Like

Long-form interviews, vendor collaborations, live stream highlights, and generic car-buying advice.

Examples of bottom performers:
- "Me-Vs-AI training platform" (6.9 vpd, 211s) — screen recording demo
- "What do us car guys look for in a vendor?" (3.8 vpd, 206s) — long vendor interview
- "What a great live and chat!" (5.5 vpd, 60s) — live stream recap

---

## Phase 5: Speech Cadence Analysis (Deepgram)

### Hypothesis

"Top-performing videos have faster WPM and less dead space (shorter gaps between words)."

### Method

Transcribed top 20 and bottom 20 videos using Deepgram with word-level timestamps. For videos without stored media URLs (older, pre-Marky), used `yt-dlp` to download from TikTok share URLs, then sent file bytes to Deepgram.

- Top 20: 20/20 transcribed successfully
- Bottom 20: 16/20 transcribed (4 failed — music-only or TikTok download issues)

### Results

| Metric | Top 20 | Bottom 20 | Delta |
|---|---|---|---|
| **WPM** | 206.2 | 206.7 | -0.4 (noise) |
| **Max gap** | 1.14s | 1.10s | +0.04s (noise) |
| **P90 gap** | 0.11s | 0.15s | -0.04s (noise) |
| **Median gap** | 0.00s | 0.00s | 0.00s |
| **Mean gap** | 0.04s | 0.04s | 0.00s |

### Conclusion

**Hypothesis rejected.** Anthony speaks at the same pace (~206 WPM) regardless of whether a video goes viral or not. Dead space is identical across both groups. His delivery style is consistent — the variable is the content, not the cadence.

---

## Phase 6: Editing Style Analysis (Gemini Video)

### Method

Used Gemini 2.5 Flash's multimodal video understanding to analyze all 40 videos (top 20 + bottom 20) second-by-second. Each video was uploaded to Gemini's file API and prompted for: cut count, camera style, text overlay frequency, b-roll presence, transitions, audio, and overall format. For videos without stored URLs, downloaded via yt-dlp first.

### Results

| Editing Factor | Top 20 | Bottom 20 | Signal? |
|---|---|---|---|
| **Text change frequency** | **1.4s avg** | **3.1s avg** | **Strong — top videos change text 2x faster** |
| Avg cuts | 1.0 | 1.3 | None |
| Text overlays present | 20/20 | 19/20 | None |
| B-roll | 3/20 | 3/20 | None |
| Music | 2/20 | 3/20 | None |
| Camera: static | 13/20 | 12/20 | None |
| Format: single take | 14/20 | 13/20 | None |
| Format: interview | 0/20 | 2/20 | Weak (interviews = bottom) |

### Key Insight: Text Chunking, Not Speaking Speed

WPM is identical (~206) across both groups, so Anthony isn't talking faster in his top videos. The difference is **how the speech is chunked into on-screen text**:

- **Top performers:** 2-3 words per text overlay, changing every ~1 second. Each overlay is a tiny phrase fragment ("MANAGERS FOCUS ON" → "NUMBERS." → "LEADERS FOCUS ON" → "PEOPLE."). Creates a rapid visual rhythm that keeps eyes locked on screen.
- **Bottom performers:** 4-8 words per overlay, changing every 3-5 seconds. Full clauses or sentences that let the viewer's eye "rest" — and that's when the thumb swipe happens.

This is the single most actionable editing insight from the entire analysis. The "karaoke effect" of rapid short-phrase text overlays that mirror speech is a measurable differentiator.

### Other Editing Observations

- His format is overwhelmingly consistent: direct-to-camera, single take, no music, no b-roll. This is true for both top and bottom performers.
- Cuts, camera movement, music, and b-roll showed zero signal — the production complexity doesn't matter.
- The only format that exclusively appears in the bottom 20 is "interview" (2/20) — long multi-person conversations.

---

## Phase 7: Content Archetype Breakdown

Manually classified all top 20 and bottom 20 videos into content archetypes based on title, subject matter, and video analysis.

### Content That Works (Top 20)

| Archetype | Count | Description |
|---|---|---|
| **Leadership principle as a question** | 10/20 | Direct-to-camera, one leadership concept framed as a question that invites comments. "Can you tell the difference between a manager and a leader?" / "How do you keep your team crystal clear?" |
| **Behind-the-scenes GM life** | 4/20 | Shows what a dealership GM actually does. Relatable and aspirational for industry insiders. "A glimpse behind the scenes of a general manager" |
| **Team recognition / culture** | 3/20 | Spotlighting team members and culture. "Let's give them some well-deserved recognition — they're the real MVPs" |
| **Specific practical advice** | 2/20 | Only works when short and punchy. "If you're trading a car in soon, watch this" (32s) |
| **Personality / fun** | 1/20 | Rare but can break through. "I'm a man of my word" (14s dance video fulfilling a bet) |

### Content That Bombs (Bottom 20)

| Archetype | Count | Description |
|---|---|---|
| **Generic car-buying advice** | 6/20 | Tips nobody asked for. "Show me the CarFax" / "Vendors are always negotiating!" |
| **Long interviews / collabs** | 3/20 | Multi-person, meandering conversations. 2+ minutes, lose focus fast. |
| **Generic motivation** | 3/20 | Vague advice without his unique angle. "Be consistent in your leadership" (60s) |
| **Live stream recaps** | 2/20 | Repurposed live content that loses context outside the live. |
| **Meta / promo content** | 2/20 | Content about content. "Follow me for tips" / "Here's what I'll be posting this month" |
| **Niche hobby / off-brand** | 2/20 | Content that doesn't match his established identity. Tattoo tours, brand nostalgia. |
| **Screen recordings / demos** | 2/20 | AI training platform demos, tech walkthroughs — wrong format for TikTok. |

### The Identity Pattern

Anthony's audience doesn't follow him for car advice — they follow him for the *identity* of a dealership GM who leads with purpose. His winning formula is first person, one clear leadership opinion, posed as a question. Not "here's a tip" but "here's how I think about X — what do you think?"

His worst content is when he steps out of that identity: generic car advice, long interviews with other people, recycled live clips, or promotional filler.

---

## Findings Summary

### What Did NOT Matter For This Creator

> **Important caveat:** These factors showed no signal for Anthony Reich's account — a car dealership GM posting direct-to-camera leadership content on TikTok. They may well matter for other creators, niches, or platforms. Day of week could matter for a restaurant. Posting hour could matter for a B2B SaaS account. Hashtag strategy could matter on Instagram. Treat these as "tested and not significant *here*" — not as universal truths. Worth re-testing for each new creator.

| Factor | How We Tested It | Evidence |
|---|---|---|
| Day of week | Qualitative (top 20 vs bottom 20) | Identical distribution across both groups |
| Posting hour | Qualitative (top 20 vs bottom 20) | Both groups post at midnight-1am (scheduled) |
| WPM / speaking pace | Deepgram word-level timestamps | 206.2 vs 206.7 WPM — 0.4 WPM delta is noise |
| Dead space (max gap) | Deepgram word-level timestamps | 1.14s vs 1.10s — identical |
| Dead space (P90 gap) | Deepgram word-level timestamps | 0.11s vs 0.15s — identical |
| Dead space (mean gap) | Deepgram word-level timestamps | 0.04s vs 0.04s — identical |
| @mentions in title | Qualitative (top 20 vs bottom 20) | 5/20 in both groups |
| CTA presence | Qualitative (top 20 vs bottom 20) | 9/20 vs 7/20 — marginal, not significant |
| Emoji in title | Feature importance (RF + GB, 5-fold CV) | No signal; model R² < 0 |
| Title word count | Feature importance (RF + GB, 5-fold CV) | 6% importance in RF but model can't generalize (R² < 0) |
| Hook quality (Gemini-rated 1-5) | Feature importance (RF + GB, 5-fold CV) | 2% importance; 36-hook sample too small, model R² < 0 |
| Hook word count | Feature importance (RF + GB, 5-fold CV) | No signal; model R² < 0 |
| Thumbnail: person pointing at camera | Qualitative (top 20 vs bottom 20) | No meaningful difference between groups |
| Thumbnail: car visible | Feature importance (RF + GB, 5-fold CV) | No signal; model R² < 0 |
| Thumbnail: person visible | Feature importance (RF + GB, 5-fold CV) | No signal; model R² < 0 |
| Posting cadence (days since last post) | Feature importance (RF + GB, 5-fold CV) | No signal; model R² < 0 |
| Hashtag usage | Feature importance (RF + GB, 5-fold CV) | No signal; model R² < 0 |
| Cut count | Gemini video analysis (top 20 vs bottom 20) | 1.0 vs 1.3 — noise |
| B-roll presence | Gemini video analysis (top 20 vs bottom 20) | 3/20 in both groups |
| Music | Gemini video analysis (top 20 vs bottom 20) | 2/20 vs 3/20 — noise |
| Camera movement | Gemini video analysis (top 20 vs bottom 20) | Static in both groups |
| Any ML-predictable metadata combination | 5-fold cross-validated RF + GB | All R² values negative (-0.19 to -1.58); worse than predicting the mean |

### What DOES Matter

| Factor | Evidence | Effect Size |
|---|---|---|
| **Duration** | Top avg 42s vs Bottom avg 68s | Large |
| **Question in title** | Top 55% vs Bottom 20% | Large |
| **Content theme (leadership)** | Top 60% vs Bottom 25% | Large |
| **Content theme (car advice)** | Top 0% vs Bottom 20% | Large (inverse) |
| **Text overlay change frequency** | Top avg 1.4s vs Bottom avg 3.1s between changes | Large |
| **Text overlay on thumbnail** | Top 100% vs Bottom 70% | Moderate |
| **Content archetype** | Leadership-as-question dominates top 20; generic advice and interviews dominate bottom 20 | Large |

---

## Actionable Recommendations

### For Anthony Reich / Huffines TikTok

1. **Keep videos under 35 seconds.** His best content is punchy — one clear point, direct-to-camera, done. The long-form interviews and live highlights consistently underperform. This doesn't mean he should never do long content, but the algorithm clearly favors his short format.

2. **Ask a question in every title.** 55% of top performers pose a direct question vs 20% of bottom performers. Questions trigger comments, which boost algorithmic reach. Formula: "What's your secret to X?" / "Can you tell the difference between X and Y?" / "How do you keep your team X?"

3. **Lean into leadership, not car tips.** His audience follows him for the GM/leadership persona. 60% of top performers are leadership content, 0% are car-buying advice. The car-buying-advice content consistently lands in the bottom. His winning identity is "leader who happens to be in the car business."

4. **Use rapid text overlays — 2-3 words at a time, changing every ~1 second.** This was the strongest editing signal. Top videos change text 2x faster (every 1.4s vs 3.1s). Short phrase fragments that mirror speech rhythm ("MANAGERS FOCUS ON" → "NUMBERS.") keep eyes on screen. Avoid full sentences in a single overlay.

5. **Always use text overlay on thumbnail.** 100% of top performers have it. It's table stakes. Every video should have a clear, bold text overlay summarizing the hook.

6. **Don't over-optimize on timing, cadence, or metadata.** Day of week, posting hour, speaking pace, and dead space showed zero signal. The algorithm's distribution decision is based on content resonance, not production mechanics.

### For Marky (Product Implications)

1. **Content type classification is more valuable than posting-time optimization.** If we build analytics, the most useful signal is content theme categorization, not "best time to post."

2. **Template-based generation should include question-in-title as a default pattern** for this type of creator.

3. **Video duration guidance** should be per-creator and per-niche, learned from their own data — not generic "best practices."

4. **Thumbnail generation** should always include text overlay for TikTok content.

---

## Strategic Product Insights for Marky

### What This Analysis Proves We Can Do

This single-creator deep dive demonstrates a pipeline that no competitor offers: pull all a creator's content, classify it, transcribe it, analyze editing style, compute feature importance, and produce a per-creator playbook — mostly automated. The moat is in making this repeatable and scalable.

### High-Value Features to Build

1. **Automated content classification** — Gemini can classify videos by archetype (leadership question, generic advice, interview, etc.) at ~$0.01/video. Run this on every client's posts automatically to build a "what works" profile per creator.

2. **Editing style fingerprinting** — Text overlay change frequency was the strongest editing signal. Extracting this from video analysis and surfacing it in creator reports ("your top posts change text every 1.4s; your bottom posts every 3.1s") is actionable and novel.

3. **Per-creator performance reports** — The format used in this analysis (top N vs bottom N comparison) is more useful than dashboards full of time-series charts. Build it as a recurring automated report.

4. **Cross-client benchmarking** — With multiple clients in the same niche (e.g., auto dealerships), compare what works across creators. "Leadership questions outperform car tips across 5 of your 7 dealership clients."

### Features to Avoid (Or Deprioritize)

1. **"Best time to post" optimization** — We tested this explicitly and found zero signal. This is the most common feature in social media tools and the least useful. It sounds good in a sales pitch but doesn't move the needle.

2. **Generic analytics dashboards** — Graphs of followers-over-time and engagement-rate don't drive action. They tell you *what happened*, not *what to do differently*. The value is in comparative analysis (top vs bottom), not trend lines.

3. **Hashtag / keyword engines** — No signal for this creator. These tools sell well but produce noise for most use cases.

### What's Missing From Current Tech Stack

| Gap | What's Needed | Why |
|---|---|---|
| Content classification at ingest | Run Gemini classification on every post when published | Enables automated "what works" reporting without manual analysis |
| Editing analysis pipeline | Integrate Gemini video analysis as a post-publish step for video content | Text change frequency is the strongest editing signal; need to capture it systematically |
| Per-creator insight model | Store "creator profile" with archetype performance data | Powers recommendations like "post more leadership questions, stop posting car tips" |
| Cross-client aggregation | Compare factor importance across multiple clients in same niche | The real moat — insights that no single client could derive alone |

### The Honest Assessment

Most of what the social media analytics industry sells is noise. Best-time-to-post, hashtag optimization, engagement rate calculators — these are features that make demos look good but don't change outcomes.

What *does* change outcomes is telling a creator "your audience responds to X, not Y, and here's the proof from your own data." That's what this analysis did manually. The product opportunity is automating it.

---

## Appendix: Methodology Details

### Data Pipeline

1. **User identification:** Queried `marky.businesses` and `marky.integrations` for the Huffines Group organization to find Anthony Reich's TikTok integration.

2. **Video fetching:** Used TikTok Content API `video/list/` endpoint with cursor-based pagination to fetch all videos. Fields: `id`, `title`, `video_description`, `duration`, `share_url`, `cover_image_url`, `like_count`, `comment_count`, `share_count`, `view_count`, `create_time`.

3. **Media URL resolution:**
   - Newer videos (posted via Marky): stored media URLs in S3/Supabase, used directly with Deepgram URL-based transcription.
   - Older videos (pre-Marky): `yt-dlp` to download from TikTok share URL, then Deepgram file-based transcription.
   - TikTok API does NOT provide direct download URLs (only `share_url` for web and `embed_link` for player). CDN URLs from `yt-dlp` are signed and reject server-side fetching (Deepgram gets 403), so local download + file upload was required.

4. **Transcription:** Deepgram nova-2 model with `smart_format=True`, `punctuate=True`, `utterances=True`. Word-level timestamps (`word.start`, `word.end`) used for gap analysis and WPM calculation.

5. **Thumbnail analysis:** Google Gemini multimodal analysis of TikTok cover images.

6. **Hook quality scoring:** Gemini rated the first 3 seconds of speech on a 1-5 scale for "hook strength" (how likely to stop a scroller).

7. **Feature importance:** scikit-learn Random Forest and Gradient Boosting with 5-fold cross-validation. Proper normalization using `views_per_day` instead of raw views.

### Tools Used

| Tool | Version/Model | Purpose |
|---|---|---|
| TikTok Content API | v2 | Video data + stats |
| Deepgram | nova-2 | Audio transcription w/ word timestamps |
| Google Gemini | 2.0-flash | Content classification, hook scoring, thumbnail analysis |
| Google Gemini | 2.5-flash | Second-by-second video editing analysis (cuts, text overlays, camera, b-roll) |
| scikit-learn | RandomForest, GradientBoosting | Feature importance |
| yt-dlp | latest | Download TikTok videos for older content |
| matplotlib | — | Feature importance and scatter plots |

### Scripts

| Script | Purpose |
|---|---|
| `scripts/tiktok_video_analysis.py` | Main data collection pipeline (TikTok API + Deepgram + Gemini) |
| `scripts/tiktok_feature_importance.py` | Phase 2 naive feature importance |
| `scripts/tiktok_enhanced_features.py` | Phase 3 enhanced features + cross-validation |
| `scripts/stored_urls.json` | Mapping of video_id → stored media URL |
| `scripts/anthony_reich_tiktok_analysis.csv` | Full dataset (185 videos) |
| `scripts/editing_analysis_results.json` | Gemini video analysis results for top 20 + bottom 20 |

### Limitations

- **Sample size:** 185 videos is sufficient for qualitative patterns but marginal for ML. Cross-validation confirmed the models can't generalize.
- **Single creator:** All findings are specific to Anthony Reich's account and audience. Patterns may differ for other creators or niches.
- **Transcript gaps:** 4 of the bottom 20 videos failed transcription (music-only clips or download failures), reducing the speech cadence sample.
- **No A/B testing:** All findings are observational correlations, not causal. We can't say "questions in titles cause more views" — only that they strongly co-occur with top performance.
- **Algorithm opacity:** TikTok's recommendation algorithm is the largest single driver of video distribution, and it's unmeasurable from outside.
