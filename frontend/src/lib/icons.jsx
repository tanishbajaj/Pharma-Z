// One drawn set, uniform 18px box and 1.6 stroke. No glyphs, no emoji.
const base = {
  width: 18, height: 18, viewBox: '0 0 18 18', fill: 'none',
  stroke: 'currentColor', strokeWidth: 1.6,
  strokeLinecap: 'round', strokeLinejoin: 'round',
}

export const IconGrid = () => (
  <svg {...base}><rect x="2" y="2" width="5.5" height="5.5" /><rect x="10.5" y="2" width="5.5" height="5.5" />
  <rect x="2" y="10.5" width="5.5" height="5.5" /><rect x="10.5" y="10.5" width="5.5" height="5.5" /></svg>
)

export const IconFlask = () => (
  <svg {...base}><path d="M7 2v5L3 14.2A1.3 1.3 0 0 0 4.2 16h9.6A1.3 1.3 0 0 0 15 14.2L11 7V2" />
  <path d="M5.8 2h6.4" /><path d="M5 11h8" /></svg>
)

export const IconBind = () => (
  <svg {...base}><circle cx="4.6" cy="9" r="2.6" /><circle cx="13.4" cy="9" r="2.6" /><path d="M7.2 9h3.6" /></svg>
)

export const IconShield = () => (
  <svg {...base}><path d="M9 1.8 15 4v4.6c0 3.7-2.5 6.5-6 7.6-3.5-1.1-6-3.9-6-7.6V4Z" /><path d="M6.6 9 8.4 11 11.6 7" /></svg>
)

export const IconFlow = () => (
  <svg {...base}><rect x="1.8" y="6.6" width="4" height="4.8" /><rect x="12.2" y="6.6" width="4" height="4.8" />
  <path d="M5.8 9h6.4" /><path d="M10.4 7.3 12.2 9l-1.8 1.7" /></svg>
)

export const IconChain = () => (
  <svg {...base}><rect x="1.8" y="3" width="6.4" height="5" /><rect x="9.8" y="10" width="6.4" height="5" />
  <path d="M5 8v3.2a1.3 1.3 0 0 0 1.3 1.3h3.5" /></svg>
)
