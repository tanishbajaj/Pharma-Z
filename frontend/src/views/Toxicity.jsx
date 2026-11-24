import { useState } from 'react'
import { api } from '../lib/api.js'
import Structure from '../lib/Structure.jsx'
import { TOX_CUT } from '../App.jsx'

const fmt = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d))

export default function Toxicity({ molecules, setMolecules }) {
  const [smiles, setSmiles] = useState('CC(C)(c1ccc(O)cc1)c1ccc(O)cc1')
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [screenBusy, setScreenBusy] = useState(false)
  const [error, setError] = useState(null)

  async function predict() {
    setBusy(true); setError(null); setResult(null)
    try {
      setResult(await api.predictToxicity(smiles))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  async function screenAll() {
    if (molecules.length === 0) return
    setScreenBusy(true); setError(null)
    try {
      const updated = [...molecules]
      for (let i = 0; i < updated.length; i++) {
        try {
          const r = await api.predictToxicity(updated[i].smiles)
          updated[i] = { ...updated[i], toxicity: r.toxicity_probability ?? null }
        } catch { updated[i] = { ...updated[i], toxicity: null } }
      }
      setMolecules(updated)
    } catch (e) { setError(e.message) } finally { setScreenBusy(false) }
  }

  const assays = result?.label_probabilities
    ? Object.entries(result.label_probabilities).sort((a, b) => b[1] - a[1])
    : []
  const screened = molecules.filter((m) => m.toxicity != null)
  const flagged = screened.filter((m) => m.toxicity > TOX_CUT).length

  return (
    <>
      <div className="work">
        <div className="head">
          <h1>Toxicity screening</h1>
          <span className="meta">Random Forest · Tox21 · 12 assays</span>
        </div>

        <div className="panel">
          <label className="field grow" style={{ minWidth: 280 }}>
            <span>SMILES</span>
            <input value={smiles} onChange={(e) => setSmiles(e.target.value)} />
          </label>
          <button className="btn" type="button" onClick={predict} disabled={busy || !smiles}>
            {busy ? 'Screening…' : 'Screen'}
          </button>
          <button
            className="btn ghost" type="button" onClick={screenAll}
            disabled={screenBusy || molecules.length === 0}
            title={molecules.length === 0 ? 'Generate candidates first' : undefined}
          >
            {screenBusy ? 'Screening…' : `Screen all ${molecules.length || ''}`.trim()}
          </button>
        </div>

        {(busy || screenBusy) && <div className="bar"><i /></div>}
        {error && <div className="note err"><strong>Screening failed.</strong> {error}</div>}

        {assays.length > 0 ? (
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th>Tox21 assay</th>
                  <th style={{ width: 300 }}>Probability</th>
                  <th style={{ width: 96 }}>Value</th>
                </tr>
              </thead>
              <tbody>
                {assays.map(([name, value]) => (
                  <tr key={name}>
                    <td className="mono">{name}</td>
                    <td>
                      <span style={{ display: 'block', height: 10, background: 'var(--ground-3)' }}>
                        <span
                          style={{
                            display: 'block', height: '100%',
                            width: `${Math.max(1, Math.min(100, value * 100))}%`,
                            background: value > 0.5 ? 'var(--tox)' : 'var(--rank)',
                          }}
                        />
                      </span>
                    </td>
                    <td className="num" style={{ fontSize: 14 }}>{fmt(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : screened.length > 0 ? (
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 54 }}>#</th>
                  <th style={{ width: 92 }}>Structure</th>
                  <th>SMILES</th>
                  <th style={{ width: 110 }}>Score</th>
                  <th style={{ width: 110 }}>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {[...screened].sort((a, b) => b.toxicity - a.toxicity).map((m, i) => (
                  <tr key={m.id}>
                    <td className="mono">{String(i + 1).padStart(3, '0')}</td>
                    <td style={{ height: 52 }}>
                      <Structure smiles={m.smiles} size={86} title={m.smiles} />
                    </td>
                    <td className="mono">{m.smiles}</td>
                    <td className="num">{fmt(m.toxicity)}</td>
                    <td>
                      <span className={`tag ${m.toxicity > TOX_CUT ? 't-tox' : 't-rank'}`}>
                        {m.toxicity > TOX_CUT ? 'Toxic' : 'Clear'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          !busy && !screenBusy && (
            <div className="empty">
              <h3>Nothing screened yet</h3>
              <p>
                Screen one molecule to see all twelve Tox21 assay probabilities,
                or screen the whole candidate set at once.
              </p>
            </div>
          )
        )}
      </div>

      <aside className="detail">
        {result ? (
          <>
            <h2>{result.is_toxic ? 'Flagged toxic' : 'Cleared'}</h2>
            <p className="sub">
              {result.aggregation || 'noisy_or'} above {result.threshold ?? TOX_CUT}
            </p>
            <div style={{ height: 112, marginBottom: 16, background: 'var(--ground-2)' }}>
              <Structure smiles={smiles} size={280} title={smiles} fallback="note" />
            </div>
            <div className="figure">
              <div className="n" style={{ color: result.is_toxic ? 'var(--tox)' : 'var(--rank)' }}>
                {fmt(result.toxicity_probability)}
              </div>
              <div className="u">Aggregated score</div>
            </div>
            <div className="rows">
              <div className="row"><span className="k">Assays</span><span className="v">{assays.length}</span></div>
              <div className="row"><span className="k">Gate</span><span className="v">&gt; {result.threshold ?? TOX_CUT}</span></div>
              <div className="row"><span className="k">Valid SMILES</span><span className="v">{result.is_valid ? 'yes' : 'no'}</span></div>
            </div>
          </>
        ) : (
          <>
            <h2>Screening set</h2>
            <p className="sub">Across the current candidates</p>
            <div className="figure">
              {screened.length
                ? <div className="n" style={{ color: flagged ? 'var(--tox)' : 'var(--ink)' }}>{flagged}</div>
                : <div className="none">Nothing screened yet</div>}
              <div className="u">Flagged toxic</div>
            </div>
            <div className="rows">
              <div className="row"><span className="k">Screened</span><span className="v">{screened.length}</span></div>
              <div className="row"><span className="k">Gate</span><span className="v">noisy-OR &gt; {TOX_CUT}</span></div>
            </div>
          </>
        )}
        <p style={{ marginTop: 18, fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.65 }}>
          Twelve independent forests, each trained only on the molecules actually
          measured for that assay. Mean ROC-AUC is 0.741 on a Bemis-Murcko
          scaffold split, 0.827 on a random split.
        </p>
      </aside>
    </>
  )
}
