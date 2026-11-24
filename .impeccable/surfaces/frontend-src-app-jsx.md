---
version: 1
slug: "frontend-src-app-jsx"
primary_target: "frontend/src/App.jsx"
related_targets: []
---

## Direction contract

THESIS: A screen of drug candidates is a periodic table of candidates. The
surface refuses the stat-cards-over-a-table dashboard, where the numbers float
free of the molecules that produced them. Here classification is the layout:
colour encodes pipeline verdict, and the grid is the argument.

OWN-WORLD: Near-black ground #12110F, bone #F2EFE6 ink, and four flat category
colours doing real jobs — chrome yellow #E4C200 generation, cobalt #1E6FD9
binding, vermilion #C8402F toxicity, green #2A7F52 ranked — one step darker
than the #2E8B5A first committed, because white ink on #2E8B5A measures
4.24:1 and the surface is read at projector distance; #2A7F52 gives 4.93:1
and the other three are verbatim. Flat, unmodulated,
zero gradients anywhere. Condensed grotesk at wall-chart density, hairline cell
rules, 6px verdict bars. Colour owns whole regions, never accents on grey.

STORY: An examiner sees 2,000 candidates classified at once, clicks a category
block to filter the grid to it, opens one cell to read its full provenance, and
watches its hash verified against the chain.

FIRST VIEWPORT: A live legend bar spans the top — four saturated category
blocks with counts, each a filter control. A 188px nav rail sits left. The body
is the candidate grid: eight columns of cells, each with index top-left, drawn
2D structure centred as the cell's symbol, formula beneath, score top-right,
and a verdict colour bar along the bottom edge. A 300px right column holds the
selected candidate's target, affinity at display scale, property rows, and
ledger hash. Primary action — Run pipeline — sits in the legend bar's right end.

FORM: Periodic table wall chart, candidate 4 of 7 on the ordered list, assigned
by the roll. seed:d7869e14bdba9187

RAISES: Identical fixed furniture in every cell, so comparison costs no eye
travel (from Character Goods Catalog, declined). Flat unmodulated colour owning
whole regions, no grain and no gradient (from Zoo & Gardens Guide Map,
declined).

FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance

## Scope

Visitor mode: Operate. Four work areas over one shell — candidate grid
(generation), binding and toxicity prediction, full pipeline run, and the
blockchain ledger. React + Vite, built to `frontend/dist/`, served by the
existing FastAPI app. All data comes from `/api/v1`; no mock data ships.

## Constraints

Inference is slow (seconds to tens of seconds) and the demo may run offline —
the 2D structure renderer must be vendored, not loaded from a CDN. Backend
returns SMILES only; structure drawing is a frontend concern.

## Unresolved

Whether `/molecules/generate-graph` is wired to a usable GraphVAE checkpoint at
demo time depends on the V2 training run finishing.
