# Brand → Music Mapping Algorithm

Turning a creator profile into a repeatable music selection system.

---

## Overview

This system converts a structured brand profile into:

1. **Archetype classification**
2. **Emotional “vibe vector”**
3. **Music cluster recommendation**
4. **Searchable genre keywords** (1–2 words)
5. **Avoid zones**

It removes subjective guessing and creates consistent music selection across creators.

---

## Step 1 — Extract Archetype Signals from the Profile

From the written brand profile, highlight repeated themes:

- **Tone words** — direct, elite, calm, intense, playful
- **Emotional outcomes** — confidence, activation, reassurance
- **Target audience state** — overwhelmed, ambitious, plateaued, rejected
- **Authority style** — prestige, relatable, coach-like, technical
- **Energy level of content** — 30-sec punch vs reflective talk

---

## Core Archetypes Used in This System

| Archetype | Signals in Profile | Emotional Promise |
|-----------|--------------------|-------------------|
| **Ruler** | Elite, curated, high standards | Status & control |
| **Sage** | Clarity, frameworks, strategy | Understanding |
| **Hero** | Breakthrough, accountability, push | Activation |
| **Caregiver** | Support, protection, guidance | Safety |
| **Everyman** | Relatable, grounded, local | Belonging |
| **Magician** | Transformation, elevation | Possibility |
| **Explorer** | Freedom, independence | Expansion |
| **Creator** | Craft, artistry, originality | Expression |

Most creators are a blend of 2–3.

---

## Step 2 — Convert to a Vibe Vector

Score each attribute from **0–1** based on profile language.

```json
{
  "authority": 0.0,
  "warmth": 0.0,
  "energy": 0.0,
  "luxury": 0.0,
  "mystery": 0.0,
  "edge": 0.0,
  "playfulness": 0.0,
  "introspection": 0.0,
  "stability": 0.0
}
```

This intermediate step prevents random genre jumping.

**Example interpretations:**

- High authority + high luxury + medium energy → **Prestige Piano**
- High warmth + high stability + low energy → **Trust Piano**
- High energy + high edge → **Motivational Pulse**
- High warmth + medium energy → **Upbeat Acoustic**

---

## Step 3 — Map Vibe Vector to Music Clusters

Standardized clusters the system should output:

---

### 1. Prestige Ethereal Piano

- **Best for:** Ruler + Sage  
- **Emotion:** Elevation, exclusivity, clarity  
- **Use when:** Leadership authority, mastermind, legacy  

**Search terms:**  
`cinematic piano`, `ambient piano`, `epic minimal`, `modern classical`

**Avoid if:** Brand is casual or action-driven.

---

### 2. Motivational Pulse

- **Best for:** Hero + Coach brands  
- **Emotion:** Activation, momentum  
- **Use when:** Accountability, punchy 30-sec clips  

**Search terms:**  
`motivational`, `corporate rock`, `trap instrumental`, `driving beat`

**Avoid if:** Brand needs calm reassurance.

---

### 3. Warm Minimal Piano (Trust Bed)

- **Best for:** Caregiver + Sage (finance, advisory)  
- **Emotion:** Stability, reassurance  
- **Use when:** Explainers, retirement, planning  

**Search terms:**  
`soft piano`, `calm corporate`, `ambient corporate`, `light piano`

**Avoid if:** Content needs urgency.

---

### 4. Upbeat Trust Pulse

- **Best for:** Caregiver + Hero (mortgage, real estate)  
- **Emotion:** Optimism + forward motion  
- **Use when:** “You can do this” content  

**Search terms:**  
`uplifting corporate`, `acoustic upbeat`, `inspirational pop`, `positive instrumental`

**Avoid if:** Brand is luxury or elite-focused.

---

### 5. Bourbon / Lounge Noir

- **Best for:** Refined masculine authority brands  
- **Emotion:** Depth, ritual, taste  
- **Use when:** Event recap, reflective leadership  

**Search terms:**  
`jazz lounge`, `loft jazz`, `smooth jazz instrumental`, `neo soul instrumental`

**Avoid if:** Brand is high-energy or young.

---

### 6. Reflective Ambient

- **Best for:** Legacy, transformation conversations  
- **Emotion:** Contemplation  
- **Use when:** Estate planning, life transitions  

**Search terms:**  
`ambient`, `atmospheric`, `minimal ambient`, `soft cinematic`

**Avoid if:** Short-form punchy content.

---

## Step 4 — Content-Length Modifier

Music must match video structure.

| Video length | Music behavior |
|--------------|----------------|
| &lt; 35 sec | Immediate identity (no slow builds) |
| 35–60 sec | Light build OK |
| 60+ sec | Can evolve gradually |

---

## Step 5 — Define Avoid Zones

Every output must include:

```json
{
  "avoid_clusters": []
}
```

This prevents:

- Ethereal on aggressive brands  
- Trap beats on financial advisors  
- Lo-fi on elite masterminds  
- Jazz on 30-sec TikTok leadership clips  

---

## Example Outputs by Creator

### Scott (Elite Mastermind)

- **Clusters:** Prestige Ethereal Piano, Lounge Noir  
- **Avoid:** Lo-fi, Motivational Trap  

**Search:**  
`cinematic piano`, `ambient piano`, `jazz lounge instrumental`

---

### Anthony (Dealership GM Leadership)

- **Clusters:** Motivational Pulse, Edgy Corporate  
- **Avoid:** Ethereal, Jazz  

**Search:**  
`motivational instrumental`, `driving corporate`, `modern rock instrumental`

---

### Wyatt (Financial Advisor)

- **Clusters:** Warm Minimal Piano, Calm Corporate  
- **Avoid:** Trap, High Energy  

**Search:**  
`soft piano`, `corporate ambient`, `light instrumental`

---

### Hunter (Mortgage Loan Officer)

- **Clusters:** Upbeat Trust Pulse, Acoustic Momentum  
- **Avoid:** Luxury Piano, Trap  

**Search:**  
`uplifting corporate`, `acoustic instrumental`, `positive pop instrumental`

---

## Final Rule

**Music must reinforce the creator’s emotional promise:**

| Promise | Cluster tendency |
|---------|-------------------|
| **Elevate** | Prestige Ethereal, Lounge Noir |
| **Activate** | Motivational Pulse |
| **Reassure** | Warm Minimal Piano, Trust Bed |
| **Enable** | Upbeat Trust Pulse, Acoustic Momentum |

If the track doesn’t strengthen that promise within **1–2 seconds**, it’s wrong.

---

## How This Connects to the Repurpose Pipeline

1. **Profile** — Read the project’s brand/profile (e.g. `01_profile.md`, `01_definition.md`).
2. **Run this algorithm** — Archetype → Vibe Vector → Clusters → Search terms + Avoid.
3. **Search** — Use `src/edit/music_search.py` with the recommended search terms (see `docs/choose_music.md`).
4. **Document** — Write results into `projects/<ClientName>/editing/music/preferences.md` (vibe, hard rules, approved tracks, avoid, search terms that worked).
5. **Use in pipeline** — `add_background_music.py` reads from the client’s `editing/music/` folder; pick a track from the approved list.
