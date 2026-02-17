# Choosing background music for a project

This doc captures the process for finding, shortlisting, and documenting approved music tracks so they can be reused across a project’s videos (e.g. repurposed shorts).

**Deriving vibe from a creator profile:** Use the [Brand → Music Mapping Algorithm](findmusic.md) to turn a brand profile into archetypes, a vibe vector, music clusters, search terms, and avoid zones. That output feeds the vibe and rules below.

## 1. Clarify vibe and rules

Before searching, agree on:

- **Vibe** — e.g. “ethereal ambient”, “motivational corporate”, “calm thought-leadership”
- **Hard rules** — e.g. no vocals, no heavy beat, tempo range, genres to avoid

These become the **Vibe** and **Hard Rules** sections in the project’s `editing/music/preferences.md`.

## 2. Search Openverse with the music_search script

From the **video-editor repo root**:

```bash
dotenvx run -f .env -- uv run python -m src.edit.music_search "<search terms>" -n 10
```

Run several searches with different tags so the stakeholder has a broad set to listen to. Examples (by vibe):

| Vibe | Search terms to try |
|------|----------------------|
| Ethereal / ambient | `ethereal ambient`, `cinematic ambient dreamy`, `atmospheric pad soft`, `soft piano ambient`, `euphoria` |
| Motivational / corporate | `motivational corporate`, `upbeat positive`, `corporate inspirational` |

Add `--json` to get machine-readable output (e.g. for scripting or pasting into a doc):

```bash
dotenvx run -f .env -- uv run python -m src.edit.music_search "ethereal ambient" -n 10 --json
```

Openverse returns **CC-licensed** tracks (Freesound, Jamendo, etc.). The script prints **direct preview/download URLs** for each result.

## 3. Share links for listening

Turn the script output into a simple list of **clickable links** (title + URL) so the stakeholder can:

- Open each track in the browser (play or download)
- Listen and decide what fits the project

No need to download at this stage; the links are enough to choose.

## 4. Stakeholder marks favorites

Stakeholder reviews the list and marks the ones they like (e.g. with a “-” or “✓” next to the title). They send back the annotated list (e.g. in chat or a doc).

## 5. Create or update project music preferences

Using the **approved** list:

1. Create the project music folder if it doesn’t exist:
   - `projects/<ClientName>/editing/music/`
2. Add or update **`preferences.md`** in that folder, following the pattern in an existing project (e.g. `projects/AnthonyReich/editing/music/preferences.md`).

**Sections to include:**

- **# Music Preferences — &lt;Client or show name&gt;**
- **Vibe** — short bullet list of the desired feel
- **Hard Rules** — no vocals, no X genre, tempo, etc.
- **Approved Tracks** — table: suggested filename, title, artist, license, duration
- **Rejected Examples** — optional; a few tracks that were listened to but not chosen (helps future searches)
- **Notes** — source (Openverse), license mix, search tags that worked
- **Download links** — direct URLs for each approved track so they can be downloaded and, if desired, saved with the suggested filenames

Suggested filename pattern for the table: `NN_short-title_artist-slug.mp3` (e.g. `01_ethereal-ambient_clacksberg.mp3`).

## 6. Optional: download into the project folder

If you want **local files** for `add_background_music` or manual editing:

- Download each approved track from the links in `preferences.md`
- Save into `projects/<ClientName>/editing/music/` using the suggested filenames from the table

Then `add_background_music` (or the pipeline) can reference that folder when adding music to a video.

## Reference: example preferences structure

See:

- `projects/AnthonyReich/editing/music/preferences.md` — motivational corporate, local filenames
- `projects/ScottJoseph/editing/music/preferences.md` — ethereal ambient, links + suggested filenames

Both use the same section layout (Vibe, Hard Rules, Approved Tracks table, Rejected Examples, Notes) so the process stays consistent across projects.
