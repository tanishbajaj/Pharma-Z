const BASE = '/api/v1'

async function call(path, options = {}) {
  let response
  try {
    response = await fetch(BASE + path, {
      headers: { 'Content-Type': 'application/json' },
      ...options,
    })
  } catch (error) {
    throw new Error(
      'Cannot reach the backend. Start it with: uvicorn app.main:app --reload'
    )
  }

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`
    try {
      const body = await response.json()
      if (body.detail) {
        detail = typeof body.detail === 'string'
          ? body.detail
          : JSON.stringify(body.detail)
      }
    } catch { /* response carried no JSON body */ }
    throw new Error(detail)
  }

  return response.json()
}

export const api = {
  health: () => call('/health'),

  generate: (count, temperature) =>
    call('/molecules/generate', {
      method: 'POST',
      body: JSON.stringify({ num_molecules: count, temperature }),
    }),

  generateGraph: (count, temperature) =>
    call('/molecules/generate-graph', {
      method: 'POST',
      body: JSON.stringify({ num_molecules: count, temperature }),
    }),

  validate: (smiles) =>
    call('/molecules/validate', {
      method: 'POST',
      body: JSON.stringify({ smiles }),
    }),

  predictBinding: (drug_smiles, protein_sequence) =>
    call('/binding/predict', {
      method: 'POST',
      body: JSON.stringify({ drug_smiles, protein_sequence }),
    }),

  batchBinding: (drug_smiles_list, protein_sequence_list) =>
    call('/binding/batch-predict', {
      method: 'POST',
      body: JSON.stringify({ drug_smiles_list, protein_sequence_list }),
    }),

  bindingModel: () => call('/binding/model-info'),

  predictToxicity: (smiles) =>
    call('/toxicity/predict', {
      method: 'POST',
      body: JSON.stringify({ smiles }),
    }),

  runPipeline: (smiles_input, protein_sequence, num_generated = 0) =>
    call('/pipeline/run', {
      method: 'POST',
      body: JSON.stringify({ smiles_input, protein_sequence, num_generated }),
    }),

  auditLogs: (limit = 50) => call(`/audit/logs?limit=${limit}`),

  storeHash: (data_hash, event_type) =>
    call('/blockchain/store-hash', {
      method: 'POST',
      body: JSON.stringify({ data_hash, event_type }),
    }),

  verifyHash: (recordId, dataHash) =>
    call(`/blockchain/verify/${recordId}?data_hash=${encodeURIComponent(dataHash)}`),

  transactions: (limit = 50) => call(`/blockchain/transactions?limit=${limit}`),
}

// GSK3B (UniProt P49841) — the type-2 diabetes kinase target used in the
// project's own screening run.
export const GSK3B =
  'MSGRPRTTSFAESCKPVQQPSAFGSMKVSRDKDGSKVTTVVATPGQGPDRPQEVSYTDTKVIGNGSFGVVYQAKLCDSG' +
  'ELVAIKKVLQDKRFKNRELQIMRKLDHCNIVRLRYFFYSSGEKKDEVYLNLVLDYVPETVYRVARHYSRAKQTLPVIYV' +
  'KLYMYQLFRSLAYIHSFGICHRDIKPQNLLLDPDTAVLKLCDFGSAKQLVRGEPNVSYICSRYYRAPELIFGATDYTSS' +
  'IDVWSAGCVLAELLLGQPIFPGDSGVDQLVEIIKVLGTPTREQIREMNPNYTEFKFPQIKAHPWTKVFRPRTPPEAIAL' +
  'CSRLLEYTPTARLTPLEACAHSFFDELRDPNVKLPNGRDTPALFNFTTQELSSNPPLATILIPPHARIQAAASTPTNAT' +
  'AASDANTGDRGQTNNAASASASNST'
