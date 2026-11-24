import { useEffect, useState } from 'react'
import { api, GSK3B } from '../lib/api.js'
import Structure from '../lib/Structure.jsx'
import { BIND_CUT } from '../App.jsx'

const fmt = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d))

export default function Binding({ molecules, setMolecules }) {
  const [smiles, setSmiles] = useState('CC(=O)Oc1ccccc1C(=O)O')
  const [protein, setProtein] = useState(GSK3B)
  const [single, setSingle] = useState(null)
  const [busy, setBusy] = useState(false)
  const [batchBusy, setBatchBusy] = useState(false)
  const [error, setError] = useState(null)
  const [model, setModel] = useState(null)

  useEffect(() => {
    api.bindingModel().then(setModel).catch(() => setModel(null))
  }, [])

  async function predict() {
    setBusy(true); setError(null); setSingle(null)
    try {
      setSingle(await api.predictBinding(smiles, protein))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  async function scoreAll() {
    if (molecules.length === 0) return
    setBatchBusy(true); setError(null)
    try {
      const list = molecules.map((m) => m.smiles)
      const data = await api.batchBinding(list, list.map(() => protein))
      const results = data.predictions || data.results || []
      setMolecules(
        molecules.map((m, i) => {
          const r = results[i]
          const value = r?.binding_affinity_score ?? r?.binding_affinity ?? null
          return { ...m, affinity: value }
        })
      )
    } catch (e) { setError(e.message) } finally { setBatchBusy(false) }
  }

  const scored = molecules.filter((m) => m.affinity != null)
  const strong = scored.filter((m) => m.affinity >= BIND_CUT).length

  return (
    <>
      <div className="work">
        <div className="head">
          <h1>Binding affinity</h1>
          <span className="meta">
            {model?.model_type
              ? `${model.model_type} · CI ${fmt(model.concordance_index ?? model.ci, 3)}`
              : 'DeepDTA on KIBA'}
          </span>
        </div>

        <div className="panel">
          <label className="field grow" style={{ minWidth: 260 }}>
            <span>Drug SMILES</span>
            <input value={smiles} onChange={(e) => setSmiles(e.target.value)} />
          </label>
          <button className="btn" type="button" onClick={predict} disabled={busy || !smiles}>
            {busy ? 'Predicting…' : 'Predict'}
          </button>
          <button
            className="btn ghost" type="button" onClick={scoreAll}
            disabled={batchBusy || molecules.length === 0}
            title={molecules.length === 0 ? 'Generate candidates first' : undefined}
          >
            {batchBusy ? 'Scoring…' : `Score all ${molecules.length || ''}`.trim()}
          </button>
        </div>

        <div className="panel" style={{ alignItems: 'stretch' }}>
          <label className="field grow">
            <span>Target protein sequence · {protein.length} residues</span>
            <textarea
              rows={3} value={protein}
              onChange={(e) => setProtein(e.target.value.replace(/\s/g, ''))}
            />
          </label>
        </div>

        {(busy || batchBusy) && <div className="bar"><i /></div>}

        {error && <div className="note err"><strong>Prediction failed.</strong> {error}</div>}

        {single && (
          <div className="note">
            <strong style={{ fontSize: 26, fontFamily: 'var(--display)' }}>
              {fmt(single.binding_affinity_score)}
            </strong>{' '}
            KIBA affinity — {single.binding_affinity_score >= BIND_CUT
              ? 'a strong binder against this target.'
              : `below the ${BIND_CUT.toFixed(1)} threshold for a strong binder.`}
          </div>
        )}

        {scored.length > 0 ? (
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 54 }}>#</th>
                  <th style={{ width: 92 }}>Structure</th>
                  <th>SMILES</th>
                  <th style={{ width: 110 }}>Affinity</th>
                  <th style={{ width: 110 }}>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {[...scored].sort((a, b) => b.affinity - a.affinity).map((m, i) => (
                  <tr key={m.id}>
                    <td className="mono">{String(i + 1).padStart(3, '0')}</td>
                    <td style={{ height: 52 }}>
                      <Structure smiles={m.smiles} size={86} title={m.smiles} />
                    </td>
                    <td className="mono">{m.smiles}</td>
                    <td className="num">{fmt(m.affinity)}</td>
                    <td>
                      <span className={`tag ${m.affinity >= BIND_CUT ? 't-bind' : 't-none'}`}>
                        {m.affinity >= BIND_CUT ? 'Strong' : 'Weak'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          !busy && !batchBusy && (
            <div className="empty">
              <h3>No batch results</h3>
              <p>
                Predict one pair above, or generate candidates and score the whole
                set against this target in a single pass.
              </p>
            </div>
          )
        )}
      </div>

      <aside className="detail">
        <h2>Target</h2>
        <p className="sub">GSK3B · UniProt P49841</p>
        <div className="figure">
          {scored.length
            ? <div className="n">{strong}</div>
            : <div className="none">Nothing scored yet</div>}
          <div className="u">Strong binders</div>
        </div>
        <div className="rows">
          <div className="row"><span className="k">Scored</span><span className="v">{scored.length}</span></div>
          <div className="row"><span className="k">Threshold</span><span className="v">≥ {BIND_CUT.toFixed(1)}</span></div>
          <div className="row"><span className="k">Scale</span><span className="v">KIBA 0 – 17.2</span></div>
          <div className="row"><span className="k">Residues</span><span className="v">{protein.length}</span></div>
        </div>
        <p style={{ marginTop: 18, fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.65 }}>
          DeepDTA encodes drug SMILES and the protein sequence through parallel
          1D convolutions. Reported Concordance Index on the KIBA held-out test
          set is 0.846 across 11,825 pairs.
        </p>
      </aside>
    </>
  )
}
