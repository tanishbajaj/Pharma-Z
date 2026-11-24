import { useCallback, useEffect, useState } from 'react'
import { api } from '../lib/api.js'

async function sha256(text) {
  const bytes = new TextEncoder().encode(text)
  const digest = await crypto.subtle.digest('SHA-256', bytes)
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('')
}

// On the screen whose whole subject is provenance, colour has to encode the
// stage a record came from — not paint every event the same.
function tagFor(eventType) {
  const e = String(eventType || '').toLowerCase()
  if (e.includes('generat')) return 't-gen'
  if (e.includes('tox')) return 't-tox'
  if (e.includes('rank') || e.includes('pipeline')) return 't-rank'
  if (e.includes('bind') || e.includes('predict')) return 't-bind'
  return 't-none'
}

const SEED = {
  event: 'pipeline_prediction',
  drug_smiles: 'CC(=O)Oc1ccccc1C(=O)O',
  protein: 'GSK3B',
  binding_affinity: 8.8778,
  toxicity_probability: 0.1556,
  is_toxic: false,
  final_score: 4.5417,
}

export default function Ledger() {
  const [records, setRecords] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [record, setRecord] = useState(JSON.stringify(SEED, null, 2))
  const [stored, setStored] = useState(null)
  const [check, setCheck] = useState(null)
  const [busy, setBusy] = useState(false)

  const load = useCallback(async () => {
    setLoading(true); setError(null)
    try {
      const data = await api.transactions(50)
      setRecords(data.transactions || [])
    } catch (e) { setError(e.message) } finally { setLoading(false) }
  }, [])

  useEffect(() => { load() }, [load])

  async function store() {
    setBusy(true); setError(null); setCheck(null)
    try {
      const hash = await sha256(record)
      const result = await api.storeHash(hash, 'prediction')
      if (result.status !== 'stored') throw new Error(result.error || result.message || 'not stored')
      setStored({ hash, recordId: result.record_id, block: result.block_number, source: record })
      await load()
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  async function verify() {
    if (!stored) return
    setBusy(true); setError(null)
    try {
      const hash = await sha256(record)
      const result = await api.verifyHash(stored.recordId, hash)
      setCheck({ valid: result.is_valid, hash, altered: record !== stored.source })
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  function tamper() {
    try {
      const edited = JSON.parse(record)
      edited.final_score = 9.9999
      edited.is_toxic = true
      setRecord(JSON.stringify(edited, null, 2))
      setCheck(null)
    } catch { /* leave unparseable text alone */ }
  }

  return (
    <>
      <div className="work">
        <div className="head">
          <h1>Audit ledger</h1>
          <span className="meta">AuditTrail.sol · Ethereum · hashes only, never raw data</span>
        </div>

        <div className="panel" style={{ alignItems: 'stretch' }}>
          <label className="field grow">
            <span>Pipeline decision to record</span>
            <textarea
              rows={10} value={record} spellCheck="false"
              onChange={(e) => { setRecord(e.target.value); setCheck(null) }}
            />
          </label>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 9, justifyContent: 'flex-end' }}>
            <button className="btn" type="button" onClick={store} disabled={busy}>
              {busy ? 'Working…' : 'Store on chain'}
            </button>
            <button className="btn ghost" type="button" onClick={tamper} disabled={!stored || busy}>
              Tamper with it
            </button>
            <button className="btn ghost" type="button" onClick={verify} disabled={!stored || busy}>
              Verify
            </button>
          </div>
        </div>

        {busy && <div className="bar"><i /></div>}
        {error && <div className="note err"><strong>Chain unavailable.</strong> {error}</div>}

        {stored && !check && (
          <div className="note">
            <strong>Recorded as #{stored.recordId}</strong> in block {stored.block}.
            <span style={{ fontFamily: 'var(--data)', fontSize: 12, display: 'block', marginTop: 6, color: 'var(--ink-3)' }}>
              {stored.hash}
            </span>
            Now edit the record above, or press “Tamper with it”, then verify.
          </div>
        )}

        {check && (
          <div className="note" style={{ borderLeftColor: check.valid ? 'var(--rank)' : 'var(--tox)', borderLeftWidth: 1 }}>
            <strong style={{ color: check.valid ? 'var(--rank)' : 'var(--tox)' }}>
              {check.valid ? 'Verified.' : 'Verification failed.'}
            </strong>{' '}
            {check.valid
              ? 'The record hashes to the value stored on chain; it has not been altered.'
              : 'This record no longer hashes to the value stored on chain. The chain rejects the altered version.'}
            <span style={{ fontFamily: 'var(--data)', fontSize: 12, display: 'block', marginTop: 8, color: 'var(--ink-3)' }}>
              on chain &nbsp;{stored.hash}<br />
              submitted&nbsp;{check.hash}
            </span>
          </div>
        )}

        {loading ? (
          <div className="empty"><h3>Reading the chain…</h3></div>
        ) : records.length > 0 ? (
          <div className="scroll">
            <table className="tbl">
              <thead>
                <tr>
                  <th style={{ width: 70 }}>Record</th>
                  <th style={{ width: 140 }}>Event</th>
                  <th>Hash</th>
                  <th style={{ width: 180 }}>Submitted by</th>
                </tr>
              </thead>
              <tbody>
                {[...records].reverse().map((r) => (
                  <tr key={r.record_id}>
                    <td className="num" style={{ fontSize: 15 }}>#{r.record_id}</td>
                    <td><span className={`tag ${tagFor(r.event_type)}`}>{r.event_type}</span></td>
                    <td className="mono">{r.data_hash}</td>
                    <td className="mono">{r.submitted_by?.slice(0, 12)}…</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty">
            <h3>Ledger empty</h3>
            <p>
              No records on chain yet. Store a decision above, or start a local
              chain with <code style={{ fontFamily: 'var(--data)' }}>npx ganache --port 8545 --deterministic</code>{' '}
              and deploy with <code style={{ fontFamily: 'var(--data)' }}>python scripts/deploy_contract.py</code>.
            </p>
          </div>
        )}
      </div>

      <aside className="detail">
        <h2>Chain</h2>
        <p className="sub">Tamper-proof decision record</p>
        <div className="figure">
          <div className="n">{records.length}</div>
          <div className="u">Records on chain</div>
        </div>
        <div className="rows">
          <div className="row"><span className="k">Contract</span><span className="v">AuditTrail</span></div>
          <div className="row"><span className="k">Digest</span><span className="v">SHA-256</span></div>
          <div className="row"><span className="k">Stored</span><span className="v">hash only</span></div>
        </div>
        <p style={{ marginTop: 18, fontSize: 13, color: 'var(--ink-3)', lineHeight: 1.65 }}>
          Only the hash of a decision is written on chain, never the molecule or
          the prediction itself. Changing any field of the record changes its
          digest, so the stored proof no longer matches and verification fails.
        </p>
      </aside>
    </>
  )
}
