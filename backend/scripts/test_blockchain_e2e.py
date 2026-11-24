"""
End-to-end test of the blockchain audit trail.

Exercises the real application code path (app.blockchain.contract_manager and
the FastAPI routes), not a bypass, so a pass here means the deployed contract
actually works through the API.

Covers report item 5.2(3): "Deploy and test the Ethereum smart contract based
audit logging system for tamper proof recording of all pipeline decisions."

Prerequisites:
    npx ganache --port 8545 --deterministic
    python scripts/deploy_contract.py

Usage:
    python scripts/test_blockchain_e2e.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.audit.hash_generator import hash_data
from app.blockchain import contract_manager
from app.config import settings

PASSED, FAILED = [], []


def check(label, condition, detail=""):
    (PASSED if condition else FAILED).append(label)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    return condition


def section(title):
    print(f"\n{'-' * 66}\n{title}\n{'-' * 66}")


def main():
    print(f"\n{'=' * 66}")
    print("Pharma-Z — Blockchain Audit Trail: End-to-End Test")
    print(f"{'=' * 66}")

    # ------------------------------------------------------------------
    section("1. Connection and configuration")
    connected = contract_manager.connect_to_blockchain()
    check("connects to the chain", connected, settings.blockchain_provider_url)
    if not connected:
        print("\n  Start a chain:  npx ganache --port 8545 --deterministic\n")
        sys.exit(1)

    check("CONTRACT_ADDRESS is configured", bool(settings.contract_address),
          settings.contract_address)
    contract = contract_manager.get_contract()
    check("contract instance loads", contract is not None)
    if contract is None:
        print("\n  Deploy first:  python scripts/deploy_contract.py\n")
        sys.exit(1)

    abi_path = contract_manager._COMPILED_ABI_PATH
    check("using the compiled ABI, not the fallback", os.path.exists(abi_path))

    starting_count = contract.functions.getRecordCount().call()
    print(f"       records already on chain: {starting_count}")

    # ------------------------------------------------------------------
    section("2. Store a real pipeline decision")
    # A realistic audit payload: one ranked drug candidate.
    decision = {
        "event": "pipeline_prediction",
        "drug_smiles": "CC(=O)Oc1ccccc1C(=O)O",
        "protein": "GSK3B",
        "binding_affinity": 8.8778,
        "toxicity_probability": 0.1556,
        "is_toxic": False,
        "final_score": 4.5417,
        "model_versions": {"binding": "deep_dta", "toxicity": "rf_masked_scaffold"},
    }
    digest = hash_data(decision)
    print(f"  SHA-256: {digest}")

    stored = contract_manager.store_hash_on_chain(digest, event_type="prediction")
    check("store_hash_on_chain succeeds", stored.get("status") == "stored",
          stored.get("error", ""))
    record_id = stored.get("record_id")
    check("returns a record_id", record_id is not None, f"id={record_id}")
    check("returns a transaction hash", bool(stored.get("transaction_hash")),
          str(stored.get("transaction_hash"))[:20] + "...")
    check("mined into a block", stored.get("block_number") is not None,
          f"block={stored.get('block_number')}")
    check("record count incremented",
          contract.functions.getRecordCount().call() == starting_count + 1)

    # ------------------------------------------------------------------
    section("3. Verify the untampered record")
    verified = contract_manager.verify_hash_on_chain(record_id, digest)
    check("verification succeeds", verified.get("status") == "verified")
    check("hash matches -> is_valid True", verified.get("is_valid") is True)
    check("stored hash equals submitted hash",
          verified.get("stored_hash") == digest)
    check("event_type round-trips", verified.get("event_type") == "prediction",
          str(verified.get("event_type")))
    check("block timestamp present", bool(verified.get("block_timestamp")))

    # ------------------------------------------------------------------
    section("4. TAMPER DETECTION (the point of the audit trail)")
    # Someone edits the stored decision after the fact: flips is_toxic and
    # inflates the score. The on-chain hash must no longer match.
    tampered = dict(decision)
    tampered["final_score"] = 9.9999
    tampered["is_toxic"] = True
    tampered_digest = hash_data(tampered)

    print(f"  original  : {digest}")
    print(f"  tampered  : {tampered_digest}")
    check("tampering changes the hash", tampered_digest != digest)

    result = contract_manager.verify_hash_on_chain(record_id, tampered_digest)
    check("tampered data FAILS verification", result.get("is_valid") is False,
          "chain rejected the altered record")

    # A single-character change must also fail.
    nudged = digest[:-1] + ("0" if digest[-1] != "0" else "1")
    nudge_result = contract_manager.verify_hash_on_chain(record_id, nudged)
    check("one-character change FAILS verification",
          nudge_result.get("is_valid") is False)

    # ------------------------------------------------------------------
    section("5. Multiple records and immutability")
    ids = [record_id]
    for index, event in enumerate(["training", "generation", "toxicity_screen"]):
        payload = {"event": event, "sequence": index, "note": f"audit {index}"}
        outcome = contract_manager.store_hash_on_chain(hash_data(payload), event)
        ids.append(outcome.get("record_id"))
    check("stored 3 more records", len([i for i in ids if i is not None]) == 4,
          f"ids={ids}")
    check("ids are sequential", ids == list(range(ids[0], ids[0] + 4)))

    # The first record must be unchanged after later writes.
    recheck = contract_manager.verify_hash_on_chain(record_id, digest)
    check("earlier record still verifies after new writes",
          recheck.get("is_valid") is True)

    # ------------------------------------------------------------------
    section("6. Reading the audit trail back")
    transactions = contract_manager.get_all_transactions(limit=100)
    check("get_all_transactions returns records", len(transactions) >= 4,
          f"{len(transactions)} records")
    if transactions:
        latest = transactions[-1]
        check("records carry hash, type and submitter",
              all(k in latest for k in ("data_hash", "event_type", "submitted_by")))
        event_types = {t["event_type"] for t in transactions}
        check("event types preserved on chain",
              {"prediction", "training", "generation"} <= event_types,
              str(sorted(event_types)))

    # ------------------------------------------------------------------
    section("7. Through the FastAPI layer")
    try:
        from fastapi.testclient import TestClient
        from app.main import app

        client = TestClient(app)

        api_payload = {"event": "api_test", "value": 42}
        api_digest = hash_data(api_payload)

        response = client.post(
            "/api/v1/blockchain/store-hash",
            json={"data_hash": api_digest, "event_type": "api_test"},
        )
        check("POST /blockchain/store-hash -> 200", response.status_code == 200,
              f"status={response.status_code}")
        body = response.json()
        check("API reports stored", body.get("status") == "stored", str(body)[:90])
        api_record = body.get("record_id")

        good = client.get(f"/api/v1/blockchain/verify/{api_record}",
                          params={"data_hash": api_digest})
        check("GET /blockchain/verify (correct hash) -> is_valid True",
              good.status_code == 200 and good.json().get("is_valid") is True)

        bad = client.get(f"/api/v1/blockchain/verify/{api_record}",
                         params={"data_hash": hash_data({"event": "forged"})})
        check("GET /blockchain/verify (forged hash) -> is_valid False",
              bad.status_code == 200 and bad.json().get("is_valid") is False)

        listing = client.get("/api/v1/blockchain/transactions", params={"limit": 10})
        check("GET /blockchain/transactions -> 200", listing.status_code == 200)
        check("transaction listing is non-empty",
              listing.json().get("total", 0) > 0,
              f"total={listing.json().get('total')}")
    except Exception as error:
        check("FastAPI layer reachable", False, repr(error)[:160])

    # ------------------------------------------------------------------
    print(f"\n{'=' * 66}")
    print(f"  PASSED: {len(PASSED)}    FAILED: {len(FAILED)}")
    if FAILED:
        for name in FAILED:
            print(f"    x {name}")
    else:
        print("  Blockchain audit trail verified end to end.")
    print(f"{'=' * 66}\n")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
