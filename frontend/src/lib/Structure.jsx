import { useEffect, useRef, useState } from 'react'
import SmilesDrawer from 'smiles-drawer'

// smiles-drawer is bundled by Vite, not fetched from a CDN, so structures
// still draw with no network at demo time.
const THEME = {
  C: '#F2EFE6', N: '#4A90E8', O: '#E85A47', S: '#E4C200',
  F: '#45B37B', CL: '#45B37B', BR: '#C98A3A', I: '#A97BD6', P: '#E4A020',
  B: '#B8B3A4', SI: '#B8B3A4', H: '#B8B3A4',
  BACKGROUND: 'transparent',
}

let drawerCache = null
function getDrawer(size) {
  const key = `${size}`
  if (!drawerCache) drawerCache = {}
  if (!drawerCache[key]) {
    drawerCache[key] = new SmilesDrawer.SvgDrawer({
      width: size,
      height: Math.round(size * 0.72),
      bondThickness: 1.1,
      bondLength: 15,
      atomVisualization: 'default',
      terminalCarbons: false,
      explicitHydrogens: false,
      compactDrawing: false,
      padding: 4,
      themes: { pz: THEME },
    })
  }
  return drawerCache[key]
}

let counter = 0

/**
 * Draws a SMILES string as a 2D skeletal structure.
 * Falls back to the SMILES text when the string cannot be parsed, so an
 * unparseable molecule is visible rather than silently blank.
 */
export default function Structure({ smiles, size = 150, title, fallback = 'text' }) {
  const ref = useRef(null)
  const [failed, setFailed] = useState(false)
  const [id] = useState(() => `pz-struct-${counter++}`)

  useEffect(() => {
    if (!ref.current || !smiles) return
    let cancelled = false
    setFailed(false)
    ref.current.innerHTML = ''

    SmilesDrawer.parse(
      smiles,
      (tree) => {
        if (cancelled || !ref.current) return
        try {
          const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
          svg.setAttribute('id', id)
          ref.current.appendChild(svg)
          // draw(data, target, themeName, weights, infoOnly) — passing a
          // boolean in the weights slot makes the library call .every() on it.
          getDrawer(size).draw(tree, id, 'pz')
          svg.removeAttribute('width')
          svg.removeAttribute('height')
          svg.style.width = '100%'
          svg.style.height = '100%'
          if (title) {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'title')
            label.textContent = title
            svg.insertBefore(label, svg.firstChild)
          }
        } catch {
          if (!cancelled) setFailed(true)
        }
      },
      // Generative models do emit unparseable SMILES. Showing the string is
      // the honest result, not an error worth shouting about.
      () => { if (!cancelled) setFailed(true) }
    )

    return () => { cancelled = true }
  }, [smiles, size, id, title])

  if (failed) {
    // 'note' is for places that already print the SMILES elsewhere; echoing
    // the same string twice in one panel is noise, not information.
    const isNote = fallback === 'note'
    const style = {
      fontFamily: isNote ? 'var(--text)' : 'var(--data)',
      fontSize: isNote ? 12 : 10,
      color: 'var(--ink-3)', lineHeight: 1.35,
      // break-all is right for an unbroken SMILES string and wrong for
      // prose, which it splits mid-word.
      wordBreak: isNote ? 'normal' : 'break-all',
      padding: '0 6px', display: 'block', textAlign: 'center',
    }
    return <span style={style}>{isNote ? 'Not drawable' : smiles}</span>
  }

  return (
    <div
      ref={ref}
      aria-label={title || `Structure of ${smiles}`}
      role="img"
      style={{ width: '100%', height: '100%', display: 'flex',
               alignItems: 'center', justifyContent: 'center' }}
    />
  )
}
