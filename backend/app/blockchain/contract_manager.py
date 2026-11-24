"""
Blockchain Contract Manager

Manages the connection to Ethereum/Ganache and interaction with
the AuditTrail smart contract via Web3.py.
"""

import json
import os
from typing import Optional

from app.config import settings
from app.audit.hash_generator import hash_data
from app.utils.logging_config import logger

# Web3 is an optional dependency
try:
    from web3 import Web3
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    logger.warning("Web3 not installed — blockchain features unavailable")


# ---------------------------------------------------------------------------
# Global State
# ---------------------------------------------------------------------------

_web3_instance: Optional[object] = None
_contract_instance: Optional[object] = None

# scripts/deploy_contract.py writes the real compiled ABI here. Prefer it over
# the hand-written fallback below, which can silently drift from the .sol.
_COMPILED_ABI_PATH = os.path.join(
    os.path.dirname(__file__), "contracts", "AuditTrail.abi.json"
)

# Hand-written fallback ABI, used only if the contract has never been compiled.
AUDIT_TRAIL_ABI = [
    {
        "inputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "_dataHash", "type": "string"},
            {"internalType": "string", "name": "_eventType", "type": "string"}
        ],
        "name": "addRecord",
        "outputs": [{"internalType": "uint256", "name": "recordId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "_recordId", "type": "uint256"}],
        "name": "getRecord",
        "outputs": [
            {"internalType": "string", "name": "dataHash", "type": "string"},
            {"internalType": "string", "name": "eventType", "type": "string"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"},
            {"internalType": "address", "name": "submittedBy", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"internalType": "uint256", "name": "_recordId", "type": "uint256"},
            {"internalType": "string", "name": "_providedHash", "type": "string"}
        ],
        "name": "verifyRecord",
        "outputs": [{"internalType": "bool", "name": "isValid", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getRecordCount",
        "outputs": [{"internalType": "uint256", "name": "count", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "recordCount",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
]


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def connect_to_blockchain() -> bool:
    """
    Connect to the Ethereum/Ganache blockchain.

    Returns:
        True if connection is successful.
    """
    global _web3_instance

    if not WEB3_AVAILABLE:
        logger.error("Web3 not installed")
        return False

    provider_url = settings.blockchain_provider_url
    _web3_instance = Web3(Web3.HTTPProvider(provider_url))

    if _web3_instance.is_connected():
        logger.info(f"Connected to blockchain at {provider_url}")
        return True
    else:
        logger.error(f"Failed to connect to blockchain at {provider_url}")
        return False


def get_web3():
    """Get or create the Web3 instance."""
    global _web3_instance
    if _web3_instance is None:
        connect_to_blockchain()
    return _web3_instance


# ---------------------------------------------------------------------------
# Contract Interaction
# ---------------------------------------------------------------------------

def load_abi() -> list:
    """The compiled ABI if the contract has been built, else the fallback."""
    if os.path.exists(_COMPILED_ABI_PATH):
        try:
            with open(_COMPILED_ABI_PATH) as handle:
                return json.load(handle)
        except Exception as error:
            logger.warning(f"Could not read compiled ABI ({error}); using fallback")
    return AUDIT_TRAIL_ABI


def get_contract():
    """Get the AuditTrail contract instance."""
    global _contract_instance

    if _contract_instance is not None:
        return _contract_instance

    web3 = get_web3()
    if web3 is None or not WEB3_AVAILABLE:
        return None

    contract_address = settings.contract_address
    if not contract_address:
        logger.warning(
            "No contract address configured. Deploy first: "
            "python scripts/deploy_contract.py"
        )
        return None

    _contract_instance = web3.eth.contract(
        address=Web3.to_checksum_address(contract_address),
        abi=load_abi(),
    )
    return _contract_instance


def store_hash_on_chain(
    data_hash: str,
    event_type: str = "prediction",
) -> dict:
    """
    Store a hash on the blockchain via the AuditTrail contract.

    Args:
        data_hash:  SHA-256 hash to store.
        event_type: Type of audit event.

    Returns:
        Dict with transaction_hash, record_id, and status.
    """
    web3 = get_web3()
    contract = get_contract()

    if contract is None or web3 is None:
        return {
            "status": "unavailable",
            "message": "Blockchain not configured or not connected",
        }

    try:
        account = web3.eth.accounts[0]

        tx_hash = contract.functions.addRecord(
            data_hash, event_type
        ).transact({"from": account})

        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        record_count = contract.functions.getRecordCount().call()

        result = {
            "status": "stored",
            "transaction_hash": receipt.transactionHash.hex(),
            "block_number": receipt.blockNumber,
            "record_id": record_count - 1,
            "data_hash": data_hash,
            "event_type": event_type,
        }

        logger.info(f"Hash stored on blockchain: record_id={result['record_id']}")
        return result

    except Exception as error:
        logger.error(f"Blockchain storage failed: {error}")
        return {"status": "failed", "error": str(error)}


def verify_hash_on_chain(record_id: int, data_hash: str) -> dict:
    """
    Verify a hash against the blockchain record.

    Args:
        record_id: The blockchain record ID.
        data_hash: The hash to verify.

    Returns:
        Dict with verification result.
    """
    contract = get_contract()

    if contract is None:
        return {"status": "unavailable", "is_valid": False}

    try:
        stored_data = contract.functions.getRecord(record_id).call()
        stored_hash = stored_data[0]
        is_valid = stored_hash == data_hash

        return {
            "status": "verified",
            "record_id": record_id,
            "is_valid": is_valid,
            "stored_hash": stored_hash,
            "provided_hash": data_hash,
            "event_type": stored_data[1],
            "block_timestamp": stored_data[2],
        }

    except Exception as error:
        logger.error(f"Blockchain verification failed: {error}")
        return {"status": "failed", "is_valid": False, "error": str(error)}


def get_all_transactions(limit: int = 100) -> list:
    """Get recent blockchain transactions (audit records)."""
    contract = get_contract()

    if contract is None:
        return []

    try:
        record_count = contract.functions.getRecordCount().call()
        transactions = []

        start = max(0, record_count - limit)
        for record_id in range(start, record_count):
            data = contract.functions.getRecord(record_id).call()
            transactions.append({
                "record_id": record_id,
                "data_hash": data[0],
                "event_type": data[1],
                "timestamp": data[2],
                "submitted_by": data[3],
            })

        return transactions

    except Exception as error:
        logger.error(f"Failed to fetch transactions: {error}")
        return []
