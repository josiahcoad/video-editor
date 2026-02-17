# Proposal: Edit Interchange for External NLE (DaVinci Resolve)

## Problem

We want to export our edit decisions (cuts, captions, title, music) along with the original video so they can be opened in an external editor for final touches—color grading, audio tweaks, text adjustments, etc.

## Why Not CapCut

- **CapCut does not support importing** XML, EDL, AAF, or any standard interchange format. It can export to FCP XML but cannot import it.
- CapCut's internal format (`draft_content.json`, `draft_info.json`) is proprietary and reverse-engineered. Programmatic generation is possible but fragile: media validation uses hashes/UUIDs/timestamps, and the format can break with any CapCut update.

## Recommendation: FCP XML for DaVinci Resolve

**Target:** DaVinci Resolve (free, professional-grade).

**Format:** Final Cut Pro 7 XML — a well-documented interchange format that Resolve imports natively.

**Workflow:**
1. Our pipeline produces an FCP XML file describing the timeline.
2. User opens the XML in DaVinci Resolve, links to original media when prompted.
3. Cuts, audio tracks, and timeline structure are reconstructed. User applies final touches.

## What We Can Export

| Edit decision    | FCP XML support | Notes                                           |
|------------------|-----------------|-------------------------------------------------|
| Cuts (in/out)    | Yes             | Clip placements on timeline                     |
| Music track      | Yes             | Audio track with offset                         |
| Captions/subtitles | Partial       | As markers or text generators; SRT also possible |
| Title text + position | Partial    | Basic text generators; position may be approximate |

## Scope (MVP)

- **Input:** Source video path + `cuts.json` (or equivalent segment metadata).
- **Output:** Single FCP XML file + optional SRT for captions.
- **Timeline:** One sequence with:
  - Video clips (cut segments from source, with correct in/out points).
  - Audio track(s): original dialogue + music (if we have music placement metadata).

## Out of Scope (Initial)

- Per-cut face crop, jump-cut edits: these are already baked into our `03_jumpcut.mp4`. The interchange would either use the **original source video** (user re-applies our cuts manually from XML) or the **pre-rendered segment** (less flexible). Decision needed.
- Complex title styling (fonts, animations): FCP XML supports basic text; advanced styling would be done in Resolve.

## Open Questions

1. **Source vs rendered:** Should the XML reference the original long-form source (user relinks cuts) or our rendered `03_jumpcut.mp4` per segment? Original = more flexible but requires our cut boundaries in the XML. Rendered = simpler but loses ability to re-cut.
2. **Music placement:** Do we store music offset/volume in `cuts.json` or elsewhere? Need to ensure we can emit it in the XML.

## References

- [FCP 7 XML interchange](https://developer.apple.com/documentation/professional_video_applications/fcp_xml_reference)
- CapCut project format: `draft_content.json` / `draft_info.json` (proprietary, not recommended as target)
- DaVinci Resolve: File → Import → Timeline → supports FCP XML
