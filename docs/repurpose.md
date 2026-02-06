# Video Repurposing Pipeline

Turn one long-form video into a full content suite: the YouTube long-form post itself, short-form clips, text posts, and image carousels — then schedule everything.

## Environment

```bash
# All scripts are run from the video-editor directory:
cd /Users/apple/Documents/code/marky/marky-app/video-editor

# Standard invocation pattern:
dotenvx run -f .env -f ../fast-backend/.env -- uv run python src/<script.py> [args]

# For scheduling (needs prod Supabase creds):
dotenvx run -f .env -f ../fast-backend/.env -f ../fast-backend/.env.prod -- uv run python src/schedule_post.py [args]
```

**Required env vars:** `DEEPGRAM_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `SUPABASE_URL`, `SUPABASE_API_KEY`, `MARKY_BUSINESS_ID`

## Output Directory

**All pipeline outputs go in `outputs/<video-stem>/` inside the video-editor directory.** Do NOT place output files next to the source video or in random locations.

```
outputs/
  <video-stem>/
    <stem>-words.json              # transcript (Phase 1)
    <stem>-utterances.json
    <stem>-transcript.txt
    cuts.json                       # proposed cuts (Phase 2)
    segment_01.mp4                  # raw cut (Phase 3)
    segment_01-titled.mp4           # with title overlay (Phase 5)
    segment_01-captioned.mp4        # with captions burned in (Phase 6) — this is the final
    segment_02.mp4
    segment_02-titled.mp4
    segment_02-captioned.mp4
    ...
```

Create the output directory at the start of the pipeline:
```bash
mkdir -p outputs/<video-stem>
```

All scripts that produce files should write to this directory. Pass transcript/output paths accordingly.

---

## Phase 0: Download Source Video (if YouTube)

If the source is a YouTube URL, download it using `yt-dlp` from the fast-backend virtualenv:

```bash
cd /Users/apple/Documents/code/marky/marky-app/fast-backend
.venv/bin/yt-dlp --cookies-from-browser safari \
  -f "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best" \
  -o "/Users/apple/Documents/code/marky/marky-app/video-editor/%(title)s.%(ext)s" \
  "<youtube-url>"
```

- **`--cookies-from-browser safari`** is usually required — YouTube blocks raw downloads with 403 errors without it.
- If downloads still fail, update yt-dlp first: `uv pip install --upgrade yt-dlp` (the fast-backend venv). YouTube frequently changes its anti-bot measures and older versions stop working.
- `yt-dlp` is installed in the fast-backend virtualenv, not video-editor. Always use `.venv/bin/yt-dlp` from the fast-backend directory.

---

## Phase 1: Transcribe

```bash
python src/get_transcript.py <video.mp4>
```

By default, transcript files are written next to the source video. **Move or copy them into `outputs/<video-stem>/`** to keep everything organized.

Produces three files:
- `<stem>-words.json` — word-level timestamps (used by propose_cuts and add_subtitles)
- `<stem>-utterances.json` — sentence-level groupings
- `<stem>-transcript.txt` — plain text

**Do this first.** Many downstream scripts can transcribe on their own, but it's wasteful to re-transcribe. Always pass existing transcripts via `--transcript` when available.

---

## Phase 2: Propose Cuts

```bash
python src/propose_cuts.py --transcript <stem>-words.json --duration 40 --tolerance 15
```

- Outputs a JSON array of segment plans to stdout. Capture it.
- Each segment has: number, summary, sections (with start/end timestamps), estimated duration.
- `--duration` is the target length per segment (default 60, but 40 works well for punchy shorts).
- `--count N` forces exactly N segments. Without it, the LLM decides based on content quality.

### Quality control (MANDATORY — stop and wait for human approval)

After running `propose_cuts`, **always stop and present a critical review to the user before proceeding.** Do not move to Phase 3 until the user approves.

**What to do:**

1. Read the proposed cuts output carefully.
2. Read the full transcript alongside the cuts to understand what was kept and what was dropped.
3. Rate the cuts honestly (e.g. 6/10, 8/10) compared to what you would have done as an experienced editor. Consider:
   - **Tightness:** Are the segments lean, or did the LLM leave in filler/verbose stretches it should have cut?
   - **Hook quality:** Does each segment open strong, or does it start with a weak/generic moment?
   - **Section granularity:** Did the LLM use enough short sections for good pacing, or is it just keeping big continuous stretches with obvious junk removed?
   - **Duration:** Is each segment the right length for its content, or is it padded/rushed?
   - **Cold viewer test:** Would each segment make sense and hook someone with zero context?
   - **Creative decisions:** Did the LLM make any smart editorial choices (cold opens, reordering for impact), or did it just mechanically remove bad parts?
4. Tell the user your rating and explain specifically what you'd change.
5. **Wait for the user's decision:**
   - **Ship it** — proceed to Phase 3 with the current cuts.
   - **Iterate** — re-run `propose_cuts` with adjusted parameters (different `--duration`, `--tolerance`, `--count`, `--prompt`, or `--model`) to improve the result. Then re-review.

**Do NOT silently accept mediocre cuts.** The propose_cuts step is the most consequential decision in the pipeline — everything downstream (titles, captions, music) is wasted if the underlying cuts are weak.

### General quality guidelines

- **Review the output.** Check that segments are 25-55s (within tolerance). Reject anything under 20s.
- The LLM sometimes splits related sub-topics into separate segments — merge them if they belong together.
- A good segment has a hook, a body, and a strong ending. If one feels like a fragment, it probably is.
- The LLM may reuse the video's intro as a hook for multiple segments. That's fine for 1-2 segments, but not all of them.
- **Prioritize quality over quantity.** 3-5 great shorts > 10 mediocre ones.

### When NOT to split into multiple segments

The LLM is biased toward producing multiple segments even when the content doesn't warrant it. Watch for these signs:

- **Single linear procedure** (e.g., a setup guide, a recipe, a walkthrough): Splitting a step-by-step process into 3 parts produces 3 fragments, not 3 shorts. Segments 2 and 3 will lack context and won't hook a cold viewer. Prefer one condensed short covering the full flow, or just post the whole video (TikTok supports up to 10 minutes).
- **Source is already short** (under 5 minutes): There may not be enough substance to split. Consider trimming filler and posting as a single short.
- **Each proposed segment fails the "cold viewer" test:** Would someone landing on THIS segment with zero context understand what's happening and want to keep watching? If not, it's a fragment, not a standalone short.
- **The segments are just chronological slices:** If segment 1 = first third, segment 2 = middle third, segment 3 = last third, the LLM didn't do any creative editorial work — it just divided by time.

**Good source material for multi-segment splits:** Multi-topic discussions, interviews with distinct questions, listicle-style content ("5 things I learned"), debates with clear position changes.

---

## Phase 3: Cut Segments

```bash
python src/apply_cuts.py <video.mp4> <output_segment.mp4> --cuts "0.8:7.06,27.54:43.81"
```

- `--cuts` takes comma-separated `start:end` timestamp ranges (in seconds).
- These come directly from the propose_cuts output (each segment's sections).
- The script concatenates the specified ranges into one continuous clip.

---

## Phase 4: Voice Isolation (optional — only when requested)

**Skip this step by default.** Only run voice isolation if the user explicitly asks for it or if the audio is noticeably noisy.

```bash
python src/voice_isolation.py <segment.mp4>
# Outputs: <segment>-isolated.mp4
```

- Uses ElevenLabs Audio Isolation API.
- Removes background noise, hum, room reverb.
- If adding music later, voice isolation helps the voice stay clean against the music track.

---

## Phase 4b: Crop to 9:16 Portrait (for short-form)

If the source video is 16:9 landscape and the target is TikTok/Reels/Shorts, crop to 9:16 **before** adding any visual overlays (titles, captions). Overlays burned into a 16:9 frame will be clipped or mis-positioned after cropping.

```bash
ffmpeg -i <segment-isolated.mp4> -vf "crop=ih*9/16:ih" \
  -c:v h264_videotoolbox -c:a copy <segment-portrait.mp4> -y
```

- This center-crops to 9:16, keeping the full height and taking the middle vertical slice.
- Works well when the speaker is centered. If the speaker is off-center, you'll need to adjust the crop offset (add `:x_offset:0` to the crop filter).
- **All subsequent steps (check_placement, add_title, add_subtitles) must run on the cropped version.** Their placement values will be completely different for 9:16 vs 16:9.

---

## Phase 5: Title Overlay

### Step 5a: Determine placement

```bash
python src/check_placement.py <segment-isolated.mp4>
```

**Run this PER video.** Different video layouts need different placements:

| Layout | Typical title height | Notes |
|--------|---------------------|-------|
| Full-screen talking head | `--height 20 --anchor bottom` | Face is centered/upper, title sits below torso |
| Split-screen (graph + face) | `--height 15 --anchor bottom` | Face is in lower half, less room to work with |
| Split-screen screen demo | Title over the screen area (top half) | The first ~10s is normally just a preamble/starting view — it's fine to cover the screen with a title card. Use black background with white text instead of the default white box. |

The script outputs `TITLE_HEIGHT_PERCENT`, `TITLE_ANCHOR`, `CAPTION_HEIGHT_PERCENT`, and reasoning.

**DO NOT use a fixed height for all videos.** The split-screen layout caught us — `--height 20` put the title right on the speaker's face because the face was much lower in the frame than expected.

**Split-screen screen demos:** For product demo videos where the top half is a screen recording and the bottom half is the speaker, the title can be placed OVER the screen area. The first ~10 seconds is typically a preamble before any meaningful on-screen action, so covering it is fine. Consider using a black background with white text for these to contrast with the screen content.

### Step 5b: Apply title

```bash
python src/add_title.py <segment-isolated.mp4> "Title Text Here" <output.mp4> \
  --duration 6 --height <N> --anchor bottom --voice voices/marky_default.md
```

- `--duration 6` — show the title for 6 seconds (not 20 — that's too long for a short).
- Title is rendered as a white rounded rectangle with black bold text.
- If no title text is provided, the script will auto-generate one from the transcript via LLM.

### Title constraints

- **Platform chrome:** TikTok/Instagram/YouTube Shorts all have UI overlays in the bottom ~10% of the frame. Never place the title there (i.e., don't go below `--height 8` or so).
- **Split-screen trade-off:** With split-screen content, the face occupies the lower half. You're stuck between overlapping the face (~15%) or getting clipped by platform chrome (~0-8%). 15% is the least-bad option.
- Title text is auto-uppercased. Keep titles concise (aim for 2 lines max at 30 chars/line).

### Title QC (MANDATORY — visual inspection)

After adding the title, **always extract a frame at ~1-2s and visually inspect it.** Check:
- Title is NOT overlapping the speaker's face
- Title is NOT clipped by platform chrome (bottom ~10%)
- Text is readable (not too small, not too many lines)
- Title text is accurate and concise

If the title placement or rendering looks bad, **tell the user what's wrong and suggest iterating on the `add_title.py` script** to fix the underlying issue — don't just re-run with different params if the script itself is producing poor results.

---

## Phase 6: Captions (Subtitles)

```bash
python src/add_subtitles.py <titled.mp4> \
  --output <captioned.mp4> \
  --height 43 \
  --replace "Markey:Marky"
```

- `--height` sets the caption vertical position (0=bottom, 100=top). Use the value from `check_placement.py` (`CAPTION_HEIGHT_PERCENT`).
- `--replace` fixes common transcription errors. Format: `"wrong1:right1,wrong2:right2"`.
- Captions are burned in as styled ASS subtitles (black text, white outline, 3 words per line).

### Caption placement guidelines

| Layout | Caption height | Why |
|--------|---------------|-----|
| Full-screen talking head | 43% | Sits on chest, clear of face and platform chrome |
| Split-screen | 45% | Sits in the gap between graph and face |

**Known issue:** In split-screen layouts, captions at 45% can still overlap the forehead during speaking sections. There's no perfect solution — the face and the content compete for the same vertical space. This is a content/framing issue, not a tools issue.

### Caption QC (MANDATORY — visual inspection)

After burning in captions, **extract frames at two moments and visually inspect:**
1. **t=3-5s** — title + captions both visible. Check they don't overlap each other or the face.
2. **t=10s** — captions only (title has faded). Check captions are clear and well-positioned.

Evaluate the caption rendering quality critically:
- Are the captions styled well (font size, weight, outline, readability)?
- Is the word timing accurate (words appear when spoken)?
- Are word replacements applied correctly (e.g. "Marquee" → "Marky")?
- Is the overall look professional or does it feel amateur?

If the captions look bad — poor styling, wrong positioning, timing issues, ugly rendering — **tell the user what's wrong and suggest iterating on the `add_subtitles.py` script** to improve the output quality. Don't just accept mediocre results.

---

## Phase 7: Background Music (optional — only when requested)

**Skip this step by default.** Only add background music if the user explicitly asks for it.

```bash
python src/add_background_music.py <captioned.mp4> <music.mp3> <final.mp4>
```

- Voice is normalized to -16 LUFS, music to -35 LUFS (voice stays dominant).
- Music starts 5 seconds into the track to skip intros.
- If you don't have a music file, use `--tags "lofi"` to search Openverse for royalty-free tracks.
- **Reuse the same music track across all shorts** from the same source video for consistency.

---

## Phase 8: Generate Copy (Title + Caption)

```bash
# For a short-form video post:
python src/write_copy.py --transcript <segment-words.json> --voice voices/marky_default.md --platform short

# For LinkedIn (text-only, 3 options):
python src/write_copy.py --transcript <words.json> --voice voices/marky_default.md --platform linkedin --count 3

# For Twitter (text-only, 5 options):
python src/write_copy.py --transcript <words.json> --voice voices/marky_default.md --platform twitter --count 5

# For a carousel caption:
python src/write_copy.py --transcript <words.json> --voice voices/marky_default.md --platform carousel
```

- Reads transcript + voice file and generates a `{"title": "...", "caption": "..."}` JSON.
- `--platform` options: `short`, `linkedin`, `twitter`, `facebook`, `carousel`.
- `--count N` generates N distinct options (each takes a different angle).
- Outputs JSON to stdout — capture it or pipe it.
- The LLM follows `voices/marky_default.md` tone rules: no buzzwords, short sentences, concrete, human.

### Quality control on copy

- Read every generated caption. Does it sound like the founder, or like an AI?
- Titles should be 2-8 words, uppercase-friendly, work as a video overlay.
- LinkedIn posts should end with a genuine question (not "What do you think?").
- Twitter posts must be under 280 characters.
- Each option should take a DIFFERENT angle — reject if they're just rephrases.

---

## Phase 9: Schedule Posts

```bash
python src/schedule_post.py \
  --video <final.mp4> \
  --title "Title Here" \
  --caption "Caption text here" \
  --publish_to tiktok instagram youtube \
  --business_id <uuid>
```

- `--video` is optional (omit for text-only posts).
- `--publish_to` specifies platforms. Valid values: `tiktok`, `instagram`, `youtube`, `linkedIn`, `facebook`, `twitter`.
- `--status` defaults to `SCHEDULED`. Other options: `NEW`, `LIKED`.
- The script uploads to Supabase Storage, creates a `business_media` record, and creates a `post`.
- **No need to pass a scheduled time** — the system auto-schedules in a queue.

---

## Phase 10: YouTube Long-Form (Title, Description, Thumbnail)

The source video itself needs to be posted to YouTube with proper packaging. Refer to `voices/marky_youtube.md` for the full guidelines.

### Title

Use `write_copy.py` to generate options, but the title needs to **pair with the thumbnail** — they tell different halves of the same story.

```bash
python src/write_copy.py --transcript <full-words.json> --voice voices/marky_default.md --platform short --count 3
```

Then manually pick/edit the best one. YouTube titles are 50-70 chars, sentence case or lowercase.

### Description

Use the reusable template pattern (see `voices/marky_youtube.md`). The only video-specific parts are:
- First line: a hook or CTA link
- Timestamps: generate from the word-level transcript

### Thumbnail (2-step Gemini generation)

Thumbnails can't be extracted from the video — they need to be composed. Use a 2-step process:

**Step 1: Generate expressive portrait**
```python
# Feed a reference frame of the speaker to Gemini Pro with an expression prompt
client.models.generate_content(
    model='gemini-3-pro-image-preview',
    contents=[
        reference_image,
        'Generate a close-up portrait of this person with [expression]. '
        'Professional headshot, clean background. Aspect ratio 3:4.',
    ],
    config=types.GenerateContentConfig(
        image_config=types.ImageConfig(aspect_ratio='3:4')
    ),
)
```

**Step 2: Compose full thumbnail**
```python
# Feed the generated portrait + describe the full layout
client.models.generate_content(
    model='gemini-3-pro-image-preview',
    contents=[
        expressive_portrait,
        'Create a YouTube thumbnail (16:9 landscape) using this person. '
        'LEFT: face cutout. RIGHT: [visual element]. '
        'BOLD TEXT: "[2-3 words]". Dark background, high contrast. '
        'Aspect ratio 16:9.',
    ],
    config=types.GenerateContentConfig(
        image_config=types.ImageConfig(aspect_ratio='16:9')
    ),
)
```

### Thumbnail + Title Pairing Rules

- Thumbnail shows the **emotion + visual hook** (face + graph + "PICK ONE")
- Title gives the **context** ("the 2 things every social platform does differently")
- They NEVER repeat each other
- Test: cover one — does the other still make you curious?

---

## Phase 12: Text Posts (LinkedIn, Twitter, Facebook)

Use `write_copy.py` to generate text-only posts from the **full video transcript** (not individual segments).

```bash
# LinkedIn: 3 long-form thought leadership posts
python src/write_copy.py --transcript <full-words.json> --voice voices/marky_default.md --platform linkedin --count 3

# Twitter: 5-10 punchy posts  
python src/write_copy.py --transcript <full-words.json> --voice voices/marky_default.md --platform twitter --count 10

# Facebook: 3 conversational posts
python src/write_copy.py --transcript <full-words.json> --voice voices/marky_default.md --platform facebook --count 3
```

Then schedule each with `src/schedule_post.py --caption "..." --publish_to <platform>` (no `--video`).

---

## Phase 13: Image Carousels (Instagram)

Use `/tmp/gen_carousel.py` (or create a similar script) to generate carousel slides.

### Key settings

- **Model:** `gemini-3-pro-image-preview` (Banana Pro) — NOT `gemini-2.5-flash-image` (Flash has bad text rendering/spelling).
- **Aspect ratio:** `3:4` for Instagram portrait.
- **Parallelism:** Generate all slides concurrently via `ThreadPoolExecutor` — cuts wall-clock time from ~160s to ~25s for 8 slides.

### Design principles

- Typography IS the design. No stock photos, no clip art.
- Clean backgrounds (soft gradients, muted tones).
- Vary font sizes, weights, and placement to emphasize key concepts.
- Color palette: warm neutrals (off-white, charcoal, muted gold/terracotta).
- Slide 1: Hook/title slide (make them want to swipe).
- Last slide: CTA — "Save this. Share it." + correct handle (`@mymarky.ai`).

### Known limitations

- Image generation models WILL misspell words sometimes. Inspect every slide.
- The model may duplicate words or hallucinate handles. Always specify exact text in the prompt.
- For pixel-perfect text, consider generating backgrounds with Gemini and overlaying text with Pillow.

### Scheduling carousels

Upload each slide image to Supabase Storage, then create a single post with all image URLs in the `media_urls` array. Set `--publish_to instagram`.

---

## Full Pipeline Summary

```
0.  Download source      yt-dlp (if YouTube URL — see Phase 0)
1.  Transcribe           get_transcript.py
2.  Propose cuts         propose_cuts.py --transcript
3.  Quality review       (agent reviews: standalone? cold open? single vs multi?)
4.  Cut segments         apply_cuts.py (one per segment; prepend cold open if applicable)
5.  Voice isolation      voice_isolation.py (OPTIONAL — only if user requests it)
5b. Crop to 9:16         ffmpeg crop (if targeting TikTok/Reels/Shorts — BEFORE overlays)
6.  Check placement      check_placement.py (on the CROPPED version — DO NOT SKIP)
7.  Add title            add_title.py --height <per-video> --anchor bottom --duration 6
8.  Add captions         add_subtitles.py --height <per-video> --replace "Markey:Marky"
9.  Add music            add_background_music.py (OPTIONAL — only if user requests it)
10. Generate copy        write_copy.py --transcript --voice voices/marky_default.md --platform short
11. Schedule shorts      schedule_post.py --video --caption --title
12. YouTube long-form    Title + description + thumbnail (see voices/marky_youtube.md)
13. Generate text posts  write_copy.py --platform linkedin/twitter/facebook → schedule_post.py
14. Generate carousel    gen_carousel.py → upload → schedule_post.py
```

---

## Agent Oversight Role

**You are not just a script runner. You are the creative director and quality gate.**

Every script in this pipeline produces output that can fail in subtle ways — wrong placement, bad phrasing, thin content, misspellings, overlapping elements. Your job is to **inspect the output of every step** before moving to the next. If something looks wrong, fix it — re-run with different parameters, regenerate, or adjust.

### What to inspect at each phase

| Phase | What to check | How to check |
|-------|--------------|--------------|
| Transcribe | Accuracy of words, especially brand names and proper nouns | Read the transcript text file. Note words that need `--replace` later. |
| Propose cuts | Segment count, durations (25-55s), cohesion, hook quality | Read the JSON output. Does each segment tell a complete story? Would YOU watch it? |
| Cut segments | Correct timestamps, no mid-sentence cuts, smooth flow | Play back (or extract frames at cut boundaries) to verify transitions. |
| Voice isolation | Clean audio, no artifacts, voice still sounds natural | Listen to a few seconds of the output. |
| Check placement | Reasonable height values for the video's layout | Review the LLM's reasoning. Does it match what you see in the frames? |
| Add title | Title is NOT on the face, NOT clipped by platform chrome, readable | **Extract frame 0-2s and visually inspect.** This is the #1 failure point. |
| Add captions | Captions are clear of face and title, words are correct | **Extract a frame at ~3-5s (when both title and captions are visible) and inspect.** |
| Add music | Voice is dominant, music is subtle, no audio clipping | Listen to the first 10-15 seconds. |
| Caption/title text | Tone matches voices/marky_default.md, no buzzwords, appropriate for platform | Read every caption. Would the founder actually say this? |
| Carousel slides | No misspellings, no duplicated words, correct handle, consistent style | **View every slide image.** Text rendering is the weakest link. |
| YouTube thumbnail | Face is recognizable, text is ≤3 bold words, high contrast, pairs with title | **View the generated image.** 2-step generation (portrait → composition) — inspect both. |
| YouTube title | 50-70 chars, creates curiosity, pairs with thumbnail (doesn't repeat it) | Read it alongside the thumbnail. Cover one — does the other still hook you? |
| Scheduling | Correct platform targeting, correct media attached | Verify the post was created with the right media_urls and publish_to. |

### Frame extraction for visual QC

Use `extract_frame.py` to grab frames at key moments:

```bash
# Frame at title appearance (t=1s)
python src/extract_frame.py <video> --time 1

# Frame with title + captions visible (t=3-5s)
python src/extract_frame.py <video> --time 4

# Frame after title disappears, captions only (t=10s)
python src/extract_frame.py <video> --time 10
```

Prints the output path to stdout (defaults to `<video_stem>_<time>s.jpg` next to the video). Use `--output /tmp/qc.jpg` for a custom path.

**Always extract and view these frames.** Do not assume the placement is correct just because the script ran without errors.

### When to re-run vs. accept

- **Re-run** if: title overlaps face, captions are unreadable, segment is too short/long, carousel slide has misspellings, caption tone is off.
- **Accept** if: minor imperfections that don't affect watchability (e.g., captions slightly overlap chin in split-screen — this is an inherent layout constraint, not a fixable bug).
- **Escalate to the user** if: you're unsure about creative direction, the source video has quality issues, or a trade-off requires human judgment (e.g., "should we skip this thin segment or merge it with another?").

---

## Cold Open Hooks

A cold open pulls a compelling soundbite from later in the video and places it at the very beginning, before the natural intro. The viewer hears something intriguing, then the video "rewinds" to the start. This is a standard technique in podcasts, documentaries, and short-form video.

### When to use a cold open

- The video contains a **standout soundbite** — a surprising claim, a counterintuitive insight, an emotional moment, or a strong opinion — that would make a viewer think "wait, what? I need to hear the rest."
- The natural opening is **weak or generic** (e.g., "Hey everyone, welcome back..."). A cold open bypasses the slow start.
- The content has a **narrative arc** where the payoff is better than the setup. Front-loading a taste of the payoff creates anticipation.

### When NOT to use a cold open

- **How-to / procedural content** (setup guides, tutorials, recipes): There's no dramatic payoff to tease. The value is the information itself, delivered linearly.
- **The speaker's tone is flat and consistent** throughout — no peaks to pull from.
- **The opening is already a strong hook** ("I lost $50K doing this one thing"). Don't compete with a good intro.
- **Every segment would get the same cold open.** If the only hookable moment is the intro, don't reuse it across all segments.

### How to implement

1. **Find the soundbite.** Read the transcript. Look for moments that are surprising, emotional, or create an open question. Good triggers: strong opinions, unexpected numbers, personal stories, "here's what nobody tells you" moments.
2. **Extract 3-8 seconds** of the soundbite. It should be a complete phrase but leave the viewer wanting context.
3. **Prepend it to the segment** using `apply_cuts.py`. Put the soundbite timestamps first in the `--cuts` list, then the rest of the segment follows.
4. **Optional: add a visual cue** (flash to white, brief text overlay like "WAIT FOR IT") at the transition point between the cold open and the real start. This signals the "rewind" to the viewer.

### Examples of good cold open candidates

- "We spent $200K on ads and the thing that actually worked cost us nothing."
- "This is the part nobody warns you about."
- "I almost didn't share this because it's embarrassing."
- A genuine laugh, a visible reaction, an emotional beat.

### Examples of bad cold open candidates

- "Today I'm gonna show you how to..." (this is just the intro)
- Generic product descriptions or feature lists
- Anything that only makes sense with prior context

---

## Decision Points for the Agent

These are places where intelligent judgment is needed (not just running scripts):

1. **How many segments to cut?** Fewer, higher-quality segments beat many thin ones. If a topic doesn't have enough substance for a 30-40s standalone short, skip it.

2. **Title placement:** Always run `check_placement.py`. The same `--height` value does NOT work across different video layouts.

3. **Caption placement:** Use the `CAPTION_HEIGHT_PERCENT` from `check_placement.py`. Split-screen layouts need different values than full-screen.

4. **Word replacements:** Check the transcript for common misheard words (brand names, proper nouns). Add them to `--replace`.

5. **Carousel quality:** Inspect generated slides for misspellings, duplicated words, and hallucinated text. Regenerate individual slides if needed.

6. **Caption writing:** Each post should have a unique caption tailored to the segment's content and the target platform. Don't use the same caption across platforms.

7. **Music selection:** Pick music that matches the energy of the content. Lofi works for most talking-head content. Reuse the same track across segments from the same source for cohesion. Music is optional — skip it if the content doesn't benefit from it.

8. **Cold open hook:** Before finalizing cuts, read the transcript for a soundbite that would hook a cold viewer. If one exists, prepend it to the segment. See the "Cold Open Hooks" section for criteria.

9. **Single vs. multi-segment:** Not every video should be split into multiple shorts. Linear procedures, short source videos (<5 min), and content without distinct sub-topics are better as a single condensed short or posted in full.

---

## Common Pitfalls

| Pitfall | Prevention |
|---------|-----------|
| Title on face (split-screen) | Run `check_placement.py` per video |
| Title clipped by platform chrome | Never go below `--height 8` with `--anchor bottom` |
| Captions say "Markey" instead of "Marky" | Use `--replace "Markey:Marky"` |
| Re-transcribing unnecessarily | Transcribe once, pass `--transcript` everywhere |
| Carousel text misspellings | Use `gemini-3-pro-image-preview`, inspect all slides |
| Wrong social handle in carousel CTA | Hardcode `@mymarky.ai` in the prompt |
| Music too loud / voice too quiet | Script handles this automatically (-16 / -35 LUFS) |
| Title shown too long | Use `--duration 6` for shorts (not the default 20) |
| Too many thin segments | Prefer quality over quantity; merge related sub-topics |
| Overlays burned before cropping | Always crop to 9:16 BEFORE adding titles/captions |
| Splitting a linear procedure into fragments | Ask: would a cold viewer understand segment 2 alone? If not, make one condensed short |
| Missing cold open opportunity | Read the transcript for standout soundbites before finalizing cuts |
