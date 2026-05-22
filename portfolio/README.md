# Portfolio — Ahmed Mostafa

A modern, creative personal-site rebuild. Pure HTML/CSS/JS (no build step), deployable directly to GitHub Pages.

## What it is

A single-page portfolio with a "Neural Engineer" theme:
- Terminal-inspired hero with rotating-title typing animation
- Animated neural-network canvas background (respects `prefers-reduced-motion`)
- Bento grid for the four flagship projects, with mouse-tracked glow
- Visualized skills (categorized cards with animated bars)
- Vertical timeline for experience and education
- Highlighted research/publication section
- Glassmorphism contact cards

## Live demo

After deployment: **https://ahmedmostafa167.github.io/**

## Local preview

```bash
cd portfolio
python -m http.server 8000
# Visit http://localhost:8000
```

Any static server works — there's no build step.

## Deploying to `AhmedMostafa167.github.io`

The portfolio lives inside this monorepo for organization, but it deploys to a separate `AhmedMostafa167.github.io` repo (so GitHub Pages serves it at the root domain).

**One command:**

```bash
cd portfolio
bash scripts/deploy.sh
```

What the script does:
1. Clones your `AhmedMostafa167.github.io` repo to a temp directory
2. Replaces its contents with this portfolio's files (preserving `LICENSE`, `.gitignore`, `CNAME` if present)
3. Commits + pushes to the default branch
4. GitHub Pages picks it up automatically — live in ~30s

Prereqs: a git identity that has push access to `AhmedMostafa167.github.io`. No `gh` CLI required.

## File tree

```
portfolio/
├── index.html              # Single-page portfolio
├── 404.html                # Themed not-found page
├── styles/
│   ├── main.css            # Design system + layout
│   ├── animations.css      # Keyframes + reveal-on-scroll
│   └── responsive.css      # Mobile-first overrides
├── scripts/
│   ├── neural-bg.js        # Canvas neural network background
│   ├── main.js             # Nav, reveals, counters, typing, glow
│   └── deploy.sh           # One-command deploy to .github.io
├── assets/
│   └── favicon.svg         # Inline SVG favicon
└── README.md
```

## Editing content

All copy lives in `index.html`. The most common edits:

| What to change | Where |
|---|---|
| Bio / about copy | `<section id="about">` |
| Project URLs / live demo links | `<section id="projects">`, inside each `.bento__card` |
| Skills percentages | `<section id="skills">`, the `--w:` CSS variables |
| Timeline entries | `<section id="experience">`, `.timeline__item` blocks |
| Email / socials | `<section id="contact">`, `.contact__card` anchors |
| CV link | `<nav>` and `<section id="contact">` (search for the Google Drive URL) |

## Design system tokens

Centralized in `styles/main.css` under `:root`. The most useful:

```css
--bg: #0a0a0f;          /* page background */
--accent: #a855f7;      /* purple (primary) */
--accent-2: #06b6d4;    /* cyan */
--accent-3: #ec4899;    /* pink */
--accent-4: #22c55e;    /* green (status) */
--grad: linear-gradient(135deg, #a855f7 0%, #06b6d4 100%);
```

Change the gradient and you re-skin the whole site.

## Accessibility

- Semantic HTML (`<nav>`, `<main>`, `<section>`, `<article>`, `<aside>`, `<ol>`)
- Skip-to-content link
- Visible focus states on all interactive elements
- `prefers-reduced-motion` disables the canvas animation, typing effect, counters, and reveal-on-scroll
- All images decorative — page works fully with screen readers
- WCAG AA color contrast for body text

## Performance

- No build step, no framework runtime, no jQuery
- ~25 KB of CSS, ~7 KB of JS (uncompressed)
- Canvas animation throttled by `requestAnimationFrame`, pauses when tab is hidden
- Fonts: subset loaded via Google Fonts with `preconnect`
- All animations use `transform`/`opacity` (compositor-only) — no layout thrash

## Browser support

Modern evergreen browsers (Chrome / Edge / Firefox / Safari 16+). Older browsers will see a graceful fallback (no canvas, no reveal animations, but all content remains readable).
