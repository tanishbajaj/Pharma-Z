---
name: Pharma-Z
description: A periodic-table wall chart for drug candidates, where colour encodes pipeline verdict.
colors:
  ground: "#12110F"
  ground-2: "#1A1916"
  ground-3: "#221F1B"
  ink: "#F2EFE6"
  ink-2: "#B8B3A4"
  ink-3: "#9D978A"
  rule: "#5A5349"
  rule-2: "#7A7163"
  gen: "#E4C200"
  bind: "#1E6FD9"
  tox: "#C8402F"
  rank: "#2A7F52"
  gen-ink: "#12110F"
  bind-ink: "#FFFFFF"
  tox-ink: "#FFFFFF"
  rank-ink: "#FFFFFF"
typography:
  display:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "62px"
    fontWeight: 700
    lineHeight: 0.95
    letterSpacing: "-0.035em"
  headline:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "27px"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.01em"
  title:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "23px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.05em"
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tabular-nums"
  label:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.5
    letterSpacing: "0.12em"
  button:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.08em"
  nav:
    fontFamily: "Archivo Narrow, Arial Narrow, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.2
    letterSpacing: "0.05em"
  data:
    fontFamily: "ui-monospace, SF Mono, Cascadia Mono, Menlo, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.6
    letterSpacing: "0.04em"
rounded:
  none: "0"
spacing:
  hair: "4px"
  xs: "6px"
  sm: "8px"
  md: "10px"
  lg: "16px"
  xl: "18px"
  gutter: "22px"
  rail: "190px"
  detail: "320px"
  legend: "74px"
components:
  legend-stage-generated:
    backgroundColor: "{colors.gen}"
    textColor: "{colors.gen-ink}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
  legend-stage-binding:
    backgroundColor: "{colors.bind}"
    textColor: "{colors.bind-ink}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
  legend-stage-toxicity:
    backgroundColor: "{colors.tox}"
    textColor: "{colors.tox-ink}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
  legend-stage-ranked:
    backgroundColor: "{colors.rank}"
    textColor: "{colors.rank-ink}"
    rounded: "{rounded.none}"
    padding: "10px 16px"
  button-primary:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    typography: "{typography.button}"
    rounded: "{rounded.none}"
    padding: "10px 20px"
  button-primary-hover:
    backgroundColor: "{colors.gen}"
    textColor: "{colors.gen-ink}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    typography: "{typography.button}"
    rounded: "{rounded.none}"
    padding: "10px 20px"
  button-ghost-hover:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
  input-field:
    backgroundColor: "{colors.ground-2}"
    textColor: "{colors.ink}"
    typography: "{typography.data}"
    rounded: "{rounded.none}"
    padding: "8px 10px"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.ink-2}"
    typography: "{typography.nav}"
    rounded: "{rounded.none}"
    padding: "11px 16px"
  nav-item-hover:
    backgroundColor: "{colors.ground-2}"
    textColor: "{colors.ink}"
  nav-item-active:
    backgroundColor: "{colors.gen}"
    textColor: "{colors.gen-ink}"
  chart-cell:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "7px 8px 13px"
    height: "132px"
    width: "112px"
  chart-cell-hover:
    backgroundColor: "{colors.ground-2}"
  chart-cell-selected:
    backgroundColor: "{colors.ground-3}"
  tag-generated:
    backgroundColor: "{colors.gen}"
    textColor: "{colors.ground}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "2px 8px"
  tag-neutral:
    backgroundColor: "{colors.ground-3}"
    textColor: "{colors.ink}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "2px 8px"
  table-header-cell:
    backgroundColor: "{colors.ground}"
    textColor: "{colors.ink-3}"
    typography: "{typography.label}"
    rounded: "{rounded.none}"
    padding: "10px 14px"
  note:
    backgroundColor: "{colors.ground-2}"
    textColor: "{colors.ink-2}"
    rounded: "{rounded.none}"
    padding: "13px 16px"
---

# Design System: Pharma-Z

## Overview

**Creative North Star: "The Periodic Table Wall Chart"**

Pharma-Z is a wall chart, not a dashboard. Two thousand candidates are
classified at once on a single near-black field, and the classification *is*
the layout: a flat category colour tells you how far a molecule travelled
through the pipeline, and the grid of cells is the argument. Nothing floats on
a card. There is no stat row hovering above a table, because that arrangement
separates the number from the molecule that produced it.

The material is unmodulated. No gradient appears anywhere in the system, no
surface is rounded, and depth is carried entirely by three near-black grounds
and a hairline grid. Colour is never an accent sprinkled on grey — each of the
four category colours fills a whole legend block edge to edge, and the same
colour reappears as a 6px bar, a stage marker, and a table tag so a viewer
learns the code once. Condensed grotesk at wall-chart density does the rest:
tight, capitalised, numeric, with tabular figures so columns of scores line up
as a reading surface.

Everything is sized for a projector in a lit room, not for a laptop at arm's
length. That single constraint set the hairline rules a step brighter than
on-screen taste would choose, pushed the muted inks up the contrast scale, and
darkened the ranked green one step so white ink on it clears 4.5:1. The surface
also ships self-contained — both typefaces and the 2D structure renderer are
bundled, because the demo may run with no network at the moment of truth.

**Key Characteristics:**

- Classification by colour: four category colours each own a pipeline stage.
- Flat and unmodulated — zero gradients, zero corner radii, no drop shadows.
- Hairline grid rules carry the chart; cells have no fill at rest.
- Condensed grotesk at wall-chart density with tabular figures throughout.
- Browser chrome is themed, not inherited: selection, scrollbar, caret, focus.
- One authored entrance moment; everything else is a 0.18s state transition.
- Self-hosted faces and a bundled renderer — the whole surface works offline.

## Colors

A near-black ground with bone ink, and four saturated category colours that do
real work rather than decorate.

### Primary

- **Chrome Yellow** (`gen`): The generation stage, and the system's single
  interactive accent. It fills the Generated legend block, the active nav rail
  item, the focus ring, the caret, the text-selection background, the progress
  sweep, and the primary button's hover. Where a control needs to say "this one
  is live", it turns yellow.

### Secondary

- **Cobalt** (`bind`): The binding stage — strong binders at KIBA ≥ 8.0. Legend
  block, verdict bar, binding tag, pipeline stage marker.
- **Vermilion** (`tox`): The toxicity stage — flagged above the 0.67 noisy-OR
  gate. Also the only error colour in the system; a failed note carries it as a
  1px left border.
- **Deep Green** (`rank`): The ranked survivors, and by extension every
  "cleared / verified" reading.

### Neutral

- **Near Black** (`ground`): The field everything sits on, and the ink used on
  the yellow legend block.
- **Charcoal** (`ground-2`): Hover fill on cells and table rows, the inset
  ground for SMILES strings and structure wells, and the input field ground.
- **Warm Charcoal** (`ground-3`): The selected cell, the live pipeline stage,
  and the neutral tag's own ground.
- **Bone** (`ink`): Primary text, the primary button's fill, and every 2px
  structural border that divides the shell.
- **Stone** (`ink-2`): Secondary text — formulas, SMILES, nav rail at rest
  (9.0:1 on ground).
- **Ash** (`ink-3`): Tertiary text — field labels, table headers, units, meta
  lines (6.5:1 on ground).
- **Hairline** (`rule`): The chart grid and row dividers.
- **Hairline Strong** (`rule-2`): Field borders, scrollbar thumb, table header
  underline, the unclassified verdict bar.

### Named Rules

**The Classification Rule.** A category colour means one pipeline stage,
everywhere it appears — legend block, 6px cell verdict bar, table tag, pipeline
stage marker. A category colour is never used for emphasis, mood, or ornament.

**The Region Rule.** Colour owns whole regions. A category colour fills a block,
a bar, or a tag; it is not an outline on a grey card or a dot beside a label.
The only non-region uses in the system are the 1px coloured left border on a
result note and a single coloured number or verdict word in the detail column.

**The Paired Ink Rule.** Every category colour has a paired ink token chosen to
clear 4.5:1 on it, and coloured regions use that pairing: near-black on yellow
(10.8:1), white on cobalt (4.85:1), vermilion (4.96:1) and green (4.93:1). The
ranked green ships at one step darker than the hue first committed precisely to
make that pairing clear.

**The Projector Rule.** Hairline rules and muted inks are set for legibility at
projector distance, not for on-screen subtlety. A rule that looks slightly too
present on a laptop is correct; one that disappears in a lit room is not.

## Typography

**Display Font:** Archivo Narrow (self-hosted woff2, variable 400–700, with
Arial Narrow as the fallback)
**Body Font:** Archivo (self-hosted woff2, variable 400–700, with system-ui)
**Data Font:** the platform monospace stack (ui-monospace / SF Mono / Cascadia
Mono / Menlo)

**Character:** Condensed grotesk doing wall-chart work — capitalised, tightly
tracked at large sizes and widely tracked at label sizes, carrying every number
and every piece of furniture. Archivo proper carries only running prose, which
is a small share of the surface. The monospace is reserved for machine strings:
SMILES, hashes, indices, raw values.

### Hierarchy

- **Display** (Archivo Narrow 700, 62px, line-height 0.95, -0.035em): The one
  headline number per detail column — an affinity, a final score, a record
  count. One per panel, never two.
- **Headline** (Archivo Narrow 700, 26–27px, -0.01em to -0.02em): The wordmark
  and the four legend counts. These are read first and from furthest away.
- **Title** (Archivo Narrow 700, 19–23px, 0.05em, uppercase): View headings and
  detail-column headings.
- **Body** (Archivo 400, 15px / 13–14px, line-height 1.5–1.65): Running prose,
  table cells, row values. Explanatory paragraphs cap at 46ch.
- **Label** (Archivo Narrow 600, 12–14px, 0.10–0.14em, uppercase): Field labels,
  table headers, units, legend block labels, tags.
- **Data** (monospace, 12–13px): SMILES strings, hashes, indices, addresses, and
  any value that must be read character by character.

### Named Rules

**The Tabular Rule.** `font-variant-numeric: tabular-nums` is set on the body
and inherited everywhere. Numbers in a column line up by digit; a score that
shifts as it updates is a bug.

**The One Big Number Rule.** The 62px display size is reserved for the single
most consequential figure in a panel. When that figure has no value yet, the
panel says so in words at 21px — an em-dash at display size reads as a white
bar, not as an absent number.

**The Machine-String Rule.** Anything the machine produced verbatim — SMILES,
SHA-256 digests, wallet addresses, assay names — is set in the data face. Prose
about them is not.

## Layout

A fixed full-viewport shell, never a scrolling page. A 74px legend bar spans
the top; a 190px nav rail sits left; the work area flexes; a 320px detail
column sits right. Only the work area and the detail column scroll, each in its
own overflow region, so the legend and rail never leave the screen.

The candidate grid is the system's spatial signature: an auto-filling grid of
112px-minimum columns (eight across the work area at 1440px) with 132px-minimum
cells, drawn as a continuous hairline lattice — the grid carries one top and
left rule, and each cell carries its right and bottom rule, so rules never
double.

Structural divisions use a 2px bone border (legend to body, rail to work, work
to detail); internal divisions use a 1px hairline. Padding rhythm is 22px for
page gutters, 16–20px for panels and cells, 8–10px inside controls.

**Responsive.** At 1100px the rail collapses to 62px icons-only and the detail
column narrows to 268px. At 820px the shell stops being a fixed viewport and
scrolls: the legend wraps so the four counts take their own full-width 2×2 row
(four across at 390px truncates the labels to meaninglessness), the nav rail
becomes a five-column icon-and-label strip above the content so every work area
is reachable without horizontal scrolling, and the detail column moves below
the work area at full width. Cells grow to a 140px minimum so structures stay
readable on a phone.

**The Fixed Furniture Rule.** Every cell carries the same four facts in the same
four positions — index top-left, score top-right, structure centred, formula
along the bottom, verdict bar on the bottom edge. A molecule that could not be
parsed still fills all five slots. Comparison across the chart must cost no eye
travel.

## Elevation & Depth

This system has no shadows. Nothing is lifted, and no surface casts anything.
Depth is tonal and linear: three near-black grounds (`ground` → `ground-2` →
`ground-3`) separate rest, hover and selection, and hairline rules do all the
structural work a shadow would otherwise do. The only `box-shadow` declarations
in the build are flat inset rings used as borders that do not shift layout.

### Ring Vocabulary

- **Selected cell** (`inset 0 0 0 2px` bone via a pseudo-element): the chosen
  candidate in the chart.
- **Active legend block** (`box-shadow: inset 0 0 0 4px ink`, plus a 2px
  `currentColor` underline): the live category filter. The block does not change
  hue to show it is on.
- **Neutral tag** (`inset 0 0 0 1px rule-2`): gives the unclassified tag an
  edge without a fill that competes with the category colours.
- **Focus** (`outline: 2px solid gen; outline-offset: 2px`): the single focus
  treatment for every interactive element.

**The Flat Material Rule.** No gradient, no drop shadow, no blur, no glow, ever.
If something needs to come forward, it changes ground tone or gains a ring.

## Shapes

Every corner in the system is square. There is no radius token because there is
no radius: buttons, inputs, tags, cells, legend blocks and notes are all
rectangles. Form language is the grid — rectangles divided by rules, with the
only curves in the system belonging to the drawn content itself (molecular bond
geometry and the 1.6-stroke icon set).

Borders are a two-weight vocabulary: 2px bone for structural shell divisions,
1px hairline for everything internal. A note or a SMILES block is marked by a
single 1px left border rather than a box.

Icons are a drawn set in a uniform 18px box at 1.6 stroke weight, round caps and
joins, `currentColor` fill none — no icon font, no emoji, no glyph characters.

## Components

### Buttons

- **Shape:** Square (0 radius), 2px border on the same colour as the fill.
- **Primary:** Bone fill with near-black ink, condensed uppercase at 15px /
  0.08em, 10px 20px padding. It is the brightest object in a panel.
- **Hover:** Fills chrome yellow with near-black ink over 0.18s.
- **Ghost:** Transparent with a bone border and bone ink; inverts to a bone fill
  on hover. Used for secondary actions (clear filter, tamper, verify, record on
  chain).
- **Disabled:** 0.45 opacity, not-allowed cursor. No colour change.

### Legend Blocks (signature)

- **Style:** Four equal flex blocks spanning the top bar, each entirely filled
  with its category colour, carrying a 26px count above a 12–14px uppercase
  label in its paired ink. Separated by 1px of ground, not by a border colour.
- **State:** Hover brightens the block 12%. Active adds a 4px inset bone ring
  and a 2px `currentColor` underline, so the filter reads without the block
  changing hue.
- **Behaviour:** Each block is a filter control over the candidate chart, and
  the counts are live.

### Candidate Cell (signature)

- **Style:** No fill at rest, 1px hairline right and bottom rules, 112×132px
  minimum, with fixed furniture in fixed positions.
- **Verdict bar:** A 6px band flush along the bottom edge in the category colour
  of the furthest stage the candidate reached; unclassified candidates get the
  strong hairline instead.
- **States:** Hover fills charcoal; selected fills warm charcoal and gains a 2px
  bone inset ring.
- **Entrance:** The `settle` animation, staggered 10ms per cell up to the 40th.

### Tags

- **Style:** Solid category colour, square, 2px 8px padding, 12px condensed
  uppercase at 0.10em. One per row, in the verdict column.
- **Neutral variant:** Warm-charcoal ground with bone ink and a 1px hairline
  inset ring — used for "not classified", which must not read as a fifth
  category.

### Inputs / Fields

- **Style:** Charcoal ground, 1px strong-hairline border, square, 8px 10px
  padding, monospace value text, with a 12px uppercase ash label stacked above
  at 0.12em.
- **Hover:** Border lifts to ash.
- **Focus:** The system focus ring — 2px chrome yellow at 2px offset. The caret
  is chrome yellow too.
- **Placeholder:** Ash.

### Tables

- **Style:** Full-width, collapsed borders, no outer frame. Header cells are
  sticky, set in 12px uppercase ash on ground with a strong-hairline underline.
  Body rows are divided by 1px hairlines and fill charcoal on hover.
- **Cell variants:** `mono` for machine strings at 12px; `num` for figures, set
  in condensed 700 at 14–16px so numbers outweigh their labels.

### Navigation

- **Style:** A left rail of five items, each a drawn 18px icon beside a 16px
  condensed uppercase label at 0.05em, in stone at rest.
- **States:** Hover fills charcoal and lifts the ink to bone. Active fills chrome
  yellow with near-black ink at weight 700 — the same accent as every other
  "live" control.
- **Responsive:** Icons only at ≤1100px; a five-column icon-over-label strip
  above the content at ≤820px.

### Structure Renderer (signature)

2D skeletal structures are drawn client-side from SMILES by a bundled renderer
at 1.1 bond thickness, 15 bond length, no terminal carbons, no explicit
hydrogens, on a transparent background — the cell's ground shows through. The
atom palette is fixed: carbon reads as bone ink, nitrogen cobalt-blue, oxygen a
warm red, sulphur chrome yellow, halogens green, and the minor elements fall
back to stone. When a generated SMILES will not parse, the slot says so in words
rather than going blank.

### States & Feedback

- **Note:** Charcoal ground, ash-stone text, 1px left border. The error variant
  swaps the border to vermilion and lifts the text to bone; verified / failed
  results swap it to green or vermilion.
- **Empty state:** Centred, ash text, a 21px uppercase condensed heading over a
  46ch paragraph that says what to do next, never a bare "no data".
- **Progress:** A 3px strong-hairline track with a 36%-wide chrome-yellow bar
  sweeping across it, shown while inference runs. Inference takes seconds to
  tens of seconds, so the bar is content, not decoration.

### Motion

**The One Entrance Rule.** `settle` is the system's only authored entrance:
0.34s on `cubic-bezier(0.16, 1, 0.3, 1)` from an already-visible default
(opacity 0.35, 5px down), staggered 10ms per cell and capped at the 40th, so a
large batch finishes arriving. Nothing else in the system animates on arrival.

Beyond it there are exactly two moving things: the looping progress sweep, which
is functional feedback while a model runs, and the 0.18s `cubic-bezier(0.22, 1,
0.36, 1)` state transition shared by every hover, filter and selection change.

A complete `prefers-reduced-motion: reduce` block removes the entrance, freezes
the sweep to a full bar, and collapses every transition to 0.01ms.

### Browser Surfaces

The browser's own chrome is part of the design. Text selection is chrome yellow
with near-black ink; the scrollbar is an 11px strong-hairline thumb inset 3px on
a ground track, lifting to ash on hover, with `scrollbar-color` and
`scrollbar-width` set for Firefox; the caret is chrome yellow; `color-scheme`
and `theme-color` are declared so the viewport chrome matches the ground.

## Do's and Don'ts

### Do:

- **Do** let a category colour mean exactly one pipeline stage, in every place
  it appears.
- **Do** fill whole regions with colour — a block, a 6px bar, a tag — rather
  than outlining or dotting.
- **Do** use the paired ink token on every coloured region: near-black on
  yellow, white on cobalt, vermilion and green.
- **Do** keep every corner square (0 radius) and every surface flat.
- **Do** carry structure with the two-weight border vocabulary: 2px bone for
  shell divisions, 1px hairline inside.
- **Do** set every machine-produced string in the monospace data face, and keep
  `tabular-nums` on everything numeric.
- **Do** reserve the 62px display size for one figure per panel, and write words
  rather than an em-dash when that figure has no value yet.
- **Do** theme the browser's own surfaces — selection, scrollbar, caret, focus
  ring — rather than inheriting defaults.
- **Do** show what is running and on how many molecules; latency is content.
- **Do** bundle every font and renderer the surface needs. The demo may be
  offline.
- **Do** keep every cell's furniture identical and in identical positions.

### Don't:

- **Don't** use a gradient, a drop shadow, a blur or a glow anywhere. Flat inset
  rings are the only `box-shadow` this system permits.
- **Don't** round a corner.
- **Don't** introduce a fifth category colour, or reuse an existing one for
  emphasis, mood or ornament.
- **Don't** set category colour as small body text or as a thin outline on a
  grey surface — at these hues that lands near 3.8:1 on the ground and the
  system's own paired-ink pairing is the correct device.
- **Don't** darken the hairline rules for on-screen subtlety; they are tuned for
  projector distance.
- **Don't** put stat cards above a table, or let a number appear detached from
  the molecule that produced it.
- **Don't** use a glyph, emoji or icon font. Icons are drawn at 18px / 1.6
  stroke.
- **Don't** add a second entrance animation, or animate anything on arrival
  other than the candidate cells.
- **Don't** fetch a font, an icon set or a renderer from a CDN.
- **Don't** show a bare empty state; say what to do next.
