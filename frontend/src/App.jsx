import { useEffect, useMemo, useState } from 'react'
import { api } from './lib/api.js'
import { IconGrid, IconBind, IconShield, IconFlow, IconChain } from './lib/icons.jsx'
import Candidates from './views/Candidates.jsx'
import Binding from './views/Binding.jsx'
import Toxicity from './views/Toxicity.jsx'
import Pipeline from './views/Pipeline.jsx'
import Ledger from './views/Ledger.jsx'

// One entry per work area. There is no separate "Generate" item: the
// Candidates view owns generation, and a second control with the same word
// on it would name two different actions.
const VIEWS = [
  { id: 'candidates', label: 'Candidates', Icon: IconGrid },
  { id: 'binding',    label: 'Binding',    Icon: IconBind },
  { id: 'toxicity',   label: 'Toxicity',   Icon: IconShield },
  { id: 'pipeline',   label: 'Pipeline',   Icon: IconFlow },
  { id: 'ledger',     label: 'Ledger',     Icon: IconChain },
]

// Thresholds mirror the backend: KIBA >= 8.0 is a strong binder, and the
// toxicity gate is the validation-chosen noisy-OR cutoff of 0.67.
export const BIND_CUT = 8.0
export const TOX_CUT = 0.67

export function verdictOf(molecule) {
  if (!molecule) return 'none'
  if (molecule.toxicity != null && molecule.toxicity > TOX_CUT) return 'tox'
  if (molecule.ranked) return 'rank'
  if (molecule.affinity != null && molecule.affinity >= BIND_CUT) return 'bind'
  return 'gen'
}

export default function App() {
  const [view, setView] = useState('candidates')
  const [molecules, setMolecules] = useState([])
  const [selected, setSelected] = useState(null)
  const [filter, setFilter] = useState(null)
  const [online, setOnline] = useState(null)

  useEffect(() => {
    api.health().then(() => setOnline(true)).catch(() => setOnline(false))
  }, [])

  const counts = useMemo(() => {
    const c = { gen: molecules.length, bind: 0, tox: 0, rank: 0 }
    for (const m of molecules) {
      if (m.affinity != null && m.affinity >= BIND_CUT) c.bind++
      if (m.toxicity != null && m.toxicity > TOX_CUT) c.tox++
      if (m.ranked) c.rank++
    }
    return c
  }, [molecules])

  const stages = [
    { id: 'gen',  cls: 's-gen',  label: 'Generated',       value: counts.gen },
    { id: 'bind', cls: 's-bind', label: `Binding ≥ ${BIND_CUT.toFixed(1)}`, value: counts.bind },
    { id: 'tox',  cls: 's-tox',  label: 'Flagged toxic',   value: counts.tox },
    { id: 'rank', cls: 's-rank', label: 'Ranked',          value: counts.rank },
  ]

  const shared = {
    molecules, setMolecules, selected, setSelected,
    filter, setFilter, counts, goto: setView,
  }

  return (
    <div className="app">
      <header className="legend">
        <div className="mark">PHARMA-Z</div>
        <div className="stages">
          {stages.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`stage ${s.cls}${filter === s.id ? ' on' : ''}`}
              onClick={() => { setFilter(filter === s.id ? null : s.id); setView('candidates') }}
              aria-pressed={filter === s.id}
              title={filter === s.id ? `Clear the ${s.label} filter` : `Show only ${s.label}`}
            >
              <span className="count">{s.value.toLocaleString()}</span>
              <span className="label">{s.label}</span>
            </button>
          ))}
        </div>
        {/* Suppressed on the Pipeline view, where the panel's own button owns
            that action — two controls with the same words did it twice. */}
        {view !== 'pipeline' && (
          <div className="legend-act">
            <button className="btn" type="button" onClick={() => setView('pipeline')}>
              Run pipeline
            </button>
          </div>
        )}
      </header>

      {online === false && (
        <div className="note err" style={{ margin: 0, borderLeft: 'none', borderBottom: '1px solid var(--rule)' }}>
          <strong>Backend unreachable.</strong> Start it with{' '}
          <code style={{ fontFamily: 'var(--data)' }}>uvicorn app.main:app --reload</code>{' '}
          in <code style={{ fontFamily: 'var(--data)' }}>backend/</code>, then reload.
        </div>
      )}

      <div className="body">
        <nav className="rail" aria-label="Sections">
          {VIEWS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              className={view === id ? 'on' : ''}
              onClick={() => setView(id)}
              aria-current={view === id ? 'page' : undefined}
              title={label}
            >
              <Icon />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        {view === 'candidates' && <Candidates {...shared} />}
        {view === 'binding'    && <Binding {...shared} />}
        {view === 'toxicity'   && <Toxicity {...shared} />}
        {view === 'pipeline'   && <Pipeline {...shared} />}
        {view === 'ledger'     && <Ledger {...shared} />}
      </div>
    </div>
  )
}
