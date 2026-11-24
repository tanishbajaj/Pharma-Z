import { useState } from 'react'
import { api, GSK3B } from '../lib/api.js'
import Structure from '../lib/Structure.jsx'
import { BIND_CUT, TOX_CUT } from '../App.jsx'

const fmt = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d))

const STAGES = [
  { key: 'validate', label: 'Validate', cls: 'v-gen' },
  { key: 'binding',  label: 'Binding',  cls: 'v-bind' },
  { key: 'toxicity', label: 'Toxicity', cls: 'v-tox' },
  { key: 'rank',     label: 'Rank',     cls: 'v-rank' },
]

export default function Pipeline() {
  const [smiles, setSmiles] = useState('CC(=O)Oc1ccccc1C(=O)O')
  const [protein, setProtein] = useState(GSK3B)
  const [generate, setGenerate] = useState(0)
  const [result, setResult] = useState(null)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(-1)
  const [error, setError] = useState(null)

  async function run() {
    setBusy(true); setError(null); setResult(null); setStage(0)
    // The backend runs these sequentially; stepping the indicator keeps the
    // wait legible rather than showing one undifferentiated spinner.
    const ticker = setInterval(() => setStage((s) => (s < 3 ? s + 1 : s)), 900)
    try {
      setResult(await api.runPipeline(smiles, protein, Number(generate)))
      setStage(3)
    } catch (e) {
      setError(e.message); setStage(-1)
    } finally {
      clearInterval(ticker); setBusy(false)
    }
  }

  const ranked = result?.ranked_candidates || []

  return (
    <>
      <div className="work">
        <div className="head">
          <h1>Full pipeline</h1>
          <span className="meta">Validate → binding → toxicity → rank</span>
        </div>

        <div className="panel">
          <label className="field grow" style={{ minWidth: 260 }}>
            <span>Input SMILES</span>
            <input value={smiles} onChange={(e) => setSmiles(e.target.value)} />
          </label>
          <label className="field">
            <span>Also generate</span>
            <input
              type="number" min="0" max="50" value={generate} style={{ width: 92 }}
              onChange={(e) => setGenerate(e.target.value)}
            />
          </label>
          <button className="btn" type="button" onClick={run} disabled={busy || !smiles}>
            {busy ? 'Running…' : 'Run pipeline'}
          </button>
        </div>

        <div className="panel" style={{ gap: 0, padding: 0, borderBottom: '1px solid var(--rule)' }}>
          {STAGES.map((s, i) => {
            const done = stage > i || (stage === 3 && result)
            const live = busy && stage === i
            return (
              <div
                key={s.key}
                style={{
                  flex: 1, minWidth: 0, padding: '14px 18px',
                  borderRight: i < 3 ? '1px solid var(--rule)' : 'none',
                  background: live ? 'var(--ground-3)' : 'transparent',
                  opacity: stage < 0 ? 0.55 : done || live ? 1 : 0.4,
                  transition: 'opacity var(--step), background var(--step)',
                  position: 'relative',
                }}
              >
                <div style={{
                  fontFamily: 'var(--display)', fontSize: 13, letterSpacing: '0.12em',
                  textTransform: 'uppercase', color: 'var(--ink-3)',
                }}>
                  Stage {i + 1}
                </div>
                <div style={{
                  fontFamily: 'var(--display)', fontSize: 20, fontWeight: 700,
                }}>
                  {s.label}
                </div>
                {(done || live) && (
                  <span className={s.cls} style={{
                    position: 'absolute', left: 0, right: 0, bottom: 0, height: 6,
                  }} />
                )}
              </div>
            )
          })}
        </div>

        {busy && <div className="bar"><i /></div>}
        {error && <div className="note err"><strong>Pipeline failed.</strong> {error}</div>}

        {result?.status === 'failed' && (
          <div className="note err"><strong>Rejected.</strong> {result.message}</div>
        )}

        {ranked.length > 0 ? (
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 54 }}>Rank</th>
                  <th style={{ width: 92 }}>Structure</th>
                  <th>SMILES</th>
                  <th style={{ width: 96 }}>Affinity</th>
                  <th style={{ width: 96 }}>Toxicity</th>
                  <th style={{ width: 96 }}>Score</th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((c, i) => (
                  <tr key={`${c.smiles}-${i}`}>
                    <td className="num" style={{ fontSize: 15 }}>{i + 1}</td>
                    <td style={{ height: 52 }}>
                      <Structure smiles={c.smiles} size={86} title={c.smiles} />
                    </td>
                    <td className="mono">{c.smiles}</td>
                    <td className="num" style={{ fontSize: 14 }}>{fmt(c.binding_affinity, 2)}</td>
                    <td className="num" style={{
                      fontSize: 14,
                      color: c.toxicity_probability > TOX_CUT ? 'var(--tox)' : 'var(--ink)',
                    }}>
                      {fmt(c.toxicity_probability)}
                    </td>
                    <td className="num" style={{ fontSize: 14 }}>{fmt(c.final_score, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          !busy && !result && !error && (
            <div className="empty">
              <h3>Pipeline idle</h3>
              <p>
                Run one molecule end to end against the target. Set “also generate”
                above zero to generate additional candidates and rank them together.
              </p>
            </div>
          )
        )}
      </div>

      <aside className="detail">
        {result?.status === 'completed' ? (
          <>
            <h2>Result</h2>
            <p className="sub">{result.canonical_smiles || smiles}</p>
            <div style={{ height: 112, marginBottom: 16, background: 'var(--ground-2)' }}>
              <Structure smiles={result.canonical_smiles || smiles} size={280} />
            </div>
            <div className="figure">
              <div className="n">{fmt(result.final_score, 2)}</div>
              <div className="u">Final score</div>
            </div>
            <div className="rows">
              <div className="row">
                <span className="k">Binding</span>
                <span className="v" style={{ color: result.binding_affinity >= BIND_CUT ? 'var(--bind)' : 'var(--ink)' }}>
                  {fmt(result.binding_affinity, 2)}
                </span>
              </div>
              <div className="row">
                <span className="k">Toxicity</span>
                <span className="v" style={{ color: result.is_toxic ? 'var(--tox)' : 'var(--rank)' }}>
                  {fmt(result.toxicity_probability)}
                </span>
              </div>
              <div className="row"><span className="k">Drug-likeness</span><span className="v">{fmt(result.drug_likeness)}</span></div>
              <div className="row"><span className="k">Verdict</span>
                <span className="v">{result.is_toxic ? 'Rejected' : 'Passed'}</span>
              </div>
            </div>
            {result.ranking_weights && (
              <p style={{ marginTop: 16, fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.65 }}>
                Weighted by binding {result.ranking_weights.binding ?? '—'}, toxicity{' '}
                {result.ranking_weights.toxicity ?? '—'}, drug-likeness{' '}
                {result.ranking_weights.drug_likeness ?? '—'}.
              </p>
            )}
          </>
        ) : (
          <>
            <h2>Target</h2>
            <p className="sub">GSK3B · UniProt P49841</p>
            <label className="field" style={{ marginBottom: 16 }}>
              <span>Protein sequence · {protein.length} residues</span>
              <textarea
                rows={6} value={protein}
                onChange={(e) => setProtein(e.target.value.replace(/\s/g, ''))}
              />
            </label>
            <p style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.65 }}>
              Glycogen synthase kinase-3 beta, a type-2 diabetes target. Each stage
              runs on the backend in sequence; the whole run is hashed and can be
              written to the ledger.
            </p>
          </>
        )}
      </aside>
    </>
  )
}
