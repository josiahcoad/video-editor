# Extract Instagram captions from the profile grid

On a profile’s grid view, post captions are stored in the **alt** of each post image. The post images live inside a `div` with class `_aagv`.

## Instructions

1. Open the Instagram profile in your browser (e.g. hunter’s profile).
2. Open DevTools (right‑click → Inspect, or Cmd+Option+I / F12).
3. Go to the **Console** tab.
4. Paste the script below and press Enter.
5. The script will log the captions and copy a JSON array to your clipboard. Paste wherever you need it (e.g. into a voice file or for analysis).

## Script (recommended): only post captions

Targets images inside `div._aagv` so you get post captions only, not icons/avatars.

```js
(function() {
  const alts = [...document.querySelectorAll('div._aagv img')]
    .map(img => img.getAttribute('alt'))
    .filter(alt => alt != null && alt.trim() !== '');

  console.log('Post captions (div._aagv img alts):', alts.length);
  console.log(alts);
  copy(JSON.stringify(alts, null, 2));
  return alts;
})();
```

## Fallback: every img alt on the page

If the class changes or you want everything (including non‑post images):

```js
(function() {
  const alts = [...document.querySelectorAll('img')]
    .map(img => img.getAttribute('alt'))
    .filter(alt => alt != null && alt.trim() !== '');

  console.log('Total img alts:', alts.length);
  console.log(alts);
  copy(JSON.stringify(alts, null, 2));
  return alts;
})();
```

## Notes

- Run the script while the profile **grid** is visible (not inside a single post).
- `_aagv` is an Instagram internal class; if IG changes their DOM, the fallback script may still work.
- Use the JSON from the clipboard to feed into voice files (e.g. `projects/HunterZier/voices/default.md`) or other tools.
