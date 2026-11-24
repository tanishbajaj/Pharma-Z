import { useEffect, useState } from 'react'
import { api } from '../lib/api.js'
import { sha256, candidateRecord } from '../lib/hash.js'
import Structure from '../lib/Structure.jsx'
import { verdictOf, BIND_CUT, TOX_CUT } from '../App.jsx'

const TARGET = 'GSK3B · UniProt P49841'

const fmt = (v, d = 2) => (v == null ? '—' : Number(v).toFixed(d))

// Always returns a line. Every cell carries the same furniture in the same
// place, so a molecule RDKit could not parse still holds its row.
function formulaOf(properties) {
  const mw = properties?.molecular_weight
  return mw != null ? `MW ${Number(mw).toFixed(1)}` : 'MW —'
}

export default function Candidates({ molecules, setMolecules, selected, setSelected, filter, setFilter }) {
  const [count, setCount] = useState(24)
  const [temperature, setTemperature] = useState(0.7)
  const [model, setModel] = useState('smiles')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function generate() {
    setBusy(true); setError(null)
    try {
      const call = model === 'graph' ? api.generateGraph : api.generate
      const data = await call(Number(count), Number(temperature))
      const list = data.molecules || data.generated_molecules || data || []
      setMolecules(
        list
          .filter((m) => m.smiles)
          .map((m, i) => ({
            id: `${Date.now()}-${i}`,
            smiles: m.smiles,
            valid: m.is_valid !== false,
            properties: m.properties || {},
            affinity: null,
            toxicity: null,
            ranked: false,
          }))
      )
      setSelected(null)
      setFilter(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setBusy(false)
    }
  }

  const shown = molecules.filter((m) => {
    if (!filter) return true
    return verdictOf(m) === filter
  })

  const current = selected != null ? molecules.find((m) => m.id === selected) : null

  // The candidate's own audit hash, so the chain is reachable from the cell
  // rather than only from the ledger's JSON box.
  const [digest, setDigest] = useState(null)
  const [chain, setChain] = useState(null)
  const [chainBusy, setChainBusy] = useState(false)

  useEffect(() => {
    setChain(null)
    if (!current) { setDigest(null); return }
    let live = true
    sha256(candidateRecord(current, TARGET)).then((h) => { if (live) setDigest(h) })
    return () => { live = false }
  }, [current?.id, current?.affinity, current?.toxicity])

  async function recordOnChain() {
    if (!digest) return
    setChainBusy(true)
    try {
      const stored = await api.storeHash(digest, 'candidate')
      if (stored.status !== 'stored') throw new Error(stored.error || 'not stored')
      const check = await api.verifyHash(stored.record_id, digest)
      setChain({ ok: check.is_valid, id: stored.record_id, block: stored.block_number })
    } catch (e) {
      setChain({ error: e.message })
    } finally {
      setChainBusy(false)
    }
  }

  return (
    <>
      <div className="work">
        <div className="panel">
          <label className="field">
            <span>Model</span>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="smiles">SMILES VAE</option>
              <option value="graph">GraphVAE</option>
            </select>
          </label>
          <label className="field">
            <span>Molecules</span>
            <input
              type="number" min="1" max="500" value={count} style={{ width: 92 }}
              onChange={(e) => setCount(e.target.value)}
            />
          </label>
          <label className="field">
            <span>Temperature</span>
            <input
              type="number" min="0.1" max="2" step="0.1" value={temperature} style={{ width: 92 }}
              onChange={(e) => setTemperature(e.target.value)}
            />
          </label>
          <button className="btn" type="button" onClick={generate} disabled={busy}>
            {busy ? 'Generating…' : 'Generate'}
          </button>
          {filter && (
            <button className="btn ghost" type="button" onClick={() => setFilter(null)}>
              Clear filter
            </button>
          )}
        </div>

        {busy && <div className="bar"><i /></div>}

        {error && (
          <div className="note err">
            <strong>Generation failed.</strong> {error}
          </div>
        )}

        {!busy && molecules.length === 0 && !error && (
          <div className="empty">
            <h3>No candidates yet</h3>
            <p>
              Generate a batch to populate the chart. Each molecule becomes a cell;
              its colour bar records how far it travelled through the pipeline.
            </p>
          </div>
        )}

        {molecules.length > 0 && shown.length === 0 && (
          <div className="empty">
            <h3>Nothing in this category</h3>
            <p>
              No candidate currently carries that verdict. Clear the filter, or run
              binding and toxicity to classify the set.
            </p>
          </div>
        )}

        {shown.length > 0 && (
          <div className="scroll">
            <div className="chart">
              {shown.map((m, i) => {
                const verdict = verdictOf(m)
                const score = m.affinity != null ? fmt(m.affinity) : fmt(m.properties?.qed_score)
                return (
                  <button
                    key={m.id}
                    type="button"
                    className={`cell${selected === m.id ? ' sel' : ''}`}
                    style={{ animationDelay: `${Math.min(i, 40) * 10}ms` }}
                    onClick={() => setSelected(selected === m.id ? null : m.id)}
                    aria-pressed={selected === m.id}
                  >
                    <span className="idx">{String(i + 1).padStart(3, '0')}</span>
                    <span className="score">{score}</span>
                    {/* note, not the raw string: at 112px the SMILES wrapped
                        to four lines and ate the cell's fixed furniture.
                        size 150 is internal SVG resolution (CSS scales it to
                        the cell) and stays above the threshold where
                        smiles-drawer computes a negative circle radius. */}
                    <span className="struct">
                      <Structure smiles={m.smiles} size={150} title={m.smiles} fallback="note" />
                    </span>
                    <span className="formula">{formulaOf(m.properties)}</span>
                    <span className={`verdict v-${verdict}`} />
                  </button>
                )
              })}
            </div>
          </div>
        )}
      </div>

      <aside className="detail">
        {current ? (
          <>
            <h2>Candidate</h2>
            <p className="sub">{TARGET}</p>
            <div style={{ height: 118, marginBottom: 16, background: 'var(--ground-2)' }}>
              <Structure smiles={current.smiles} size={280} title={current.smiles} fallback="note" />
            </div>
            {current.affinity != null && (
              <div className="figure">
                <div className="n">{fmt(current.affinity)}</div>
                <div className="u">KIBA affinity</div>
              </div>
            )}
            <div className="smiles">{current.smiles}</div>
            <div className="rows" style={{ marginTop: 16 }}>
              {[
                ['Molecular weight', fmt(current.properties?.molecular_weight, 1)],
                ['LogP', fmt(current.properties?.logp)],
                ['TPSA', fmt(current.properties?.tpsa, 1)],
                ['QED', fmt(current.properties?.qed_score, 3)],
                ['H-donors', current.properties?.num_h_donors ?? '—'],
                ['H-acceptors', current.properties?.num_h_acceptors ?? '—'],
                ['Toxicity', current.toxicity != null ? fmt(current.toxicity, 3) : '—'],
              ].map(([k, v]) => (
                <div className="row" key={k}>
                  <span className="k">{k}</span>
                  <span className="v">{v}</span>
                </div>
              ))}
            </div>
            <div style={{ marginTop: 18 }}>
              <div style={{
                fontFamily: 'var(--display)', fontSize: 12, letterSpacing: '0.12em',
                textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 6,
              }}>
                Audit hash
              </div>
              <div className="smiles">{digest || 'computing…'}</div>
              <button
                className="btn ghost" type="button" onClick={recordOnChain}
                disabled={chainBusy || !digest}
                style={{ marginTop: 10, width: '100%', fontSize: 13, padding: '9px 10px' }}
              >
                {chainBusy ? 'Recording…' : 'Record on chain'}
              </button>
              {chain && (
                <p style={{
                  marginTop: 9, fontSize: 13, lineHeight: 1.55,
                  color: chain.error || !chain.ok ? 'var(--tox)' : 'var(--rank)',
                }}>
                  {chain.error
                    ? `Chain unavailable: ${chain.error}`
                    : chain.ok
                      ? `Verified as record #${chain.id} in block ${chain.block}.`
                      : `Stored as #${chain.id} but verification failed.`}
                </p>
              )}
            </div>

            <p style={{ marginTop: 16, fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.6 }}>
              Strong binder at KIBA ≥ {BIND_CUT.toFixed(1)}. Toxic above {TOX_CUT} on the
              noisy-OR gate chosen on a validation split.
            </p>
          </>
        ) : (
          <>
            <h2>No selection</h2>
            <p className="sub">Choose a cell to read its full record.</p>
            <p style={{ fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.7 }}>
              Each cell carries the same four facts in the same four positions:
              index, score, structure, and molecular weight. The bar along the
              bottom edge encodes the furthest stage the candidate reached.
            </p>
          </>
        )}
      </aside>
    </>
  )
}
