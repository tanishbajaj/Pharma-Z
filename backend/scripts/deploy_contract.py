"""
Compile and deploy the AuditTrail smart contract.

Closes report item 5.2(3): "Deploy and test the Ethereum smart contract based
audit logging system for tamper proof recording of all pipeline decisions."

Prerequisites — start a local chain first, in a separate terminal:
    npx ganache --port 8545 --deterministic

Usage:
    python scripts/deploy_contract.py              # compile + deploy + write .env
    python scripts/deploy_contract.py --no-env     # don't touch .env
    python scripts/deploy_contract.py --abi-only   # just regenerate the ABI

Writes the deployed address into backend/.env as CONTRACT_ADDRESS, and the
compiled ABI to app/blockchain/contracts/AuditTrail.abi.json so that
contract_manager.py can load the real ABI instead of its hand-written copy.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import solcx
from web3 import Web3

from app.config import settings

SOLC_VERSION = "0.8.19"
CONTRACT_DIR = os.path.join(os.path.dirname(__file__), "..", "app", "blockchain", "contracts")
SOURCE_PATH = os.path.join(CONTRACT_DIR, "AuditTrail.sol")
ABI_PATH = os.path.join(CONTRACT_DIR, "AuditTrail.abi.json")
ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")


def compile_contract():
    """Compile AuditTrail.sol and return (abi, bytecode)."""
    if SOLC_VERSION not in [str(v) for v in solcx.get_installed_solc_versions()]:
        print(f"  Installing solc {SOLC_VERSION} ...")
        solcx.install_solc(SOLC_VERSION)

    with open(SOURCE_PATH) as handle:
        source = handle.read()

    compiled = solcx.compile_source(
        source,
        output_values=["abi", "bin"],
        solc_version=SOLC_VERSION,
    )
    # compile_source keys look like "<stdin>:AuditTrail"
    key = next(k for k in compiled if k.endswith(":AuditTrail"))
    interface = compiled[key]
    return interface["abi"], interface["bin"]


def write_env(address):
    """Set CONTRACT_ADDRESS in backend/.env, preserving everything else."""
    lines, found = [], False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as handle:
            lines = handle.read().splitlines()

    for index, line in enumerate(lines):
        if line.startswith("CONTRACT_ADDRESS="):
            lines[index] = f"CONTRACT_ADDRESS={address}"
            found = True
            break
    if not found:
        lines.append(f"CONTRACT_ADDRESS={address}")

    with open(ENV_PATH, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Deploy the AuditTrail contract")
    parser.add_argument("--provider", default=settings.blockchain_provider_url)
    parser.add_argument("--no-env", action="store_true",
                        help="Do not write CONTRACT_ADDRESS into .env")
    parser.add_argument("--abi-only", action="store_true",
                        help="Compile and write the ABI, skip deployment")
    args = parser.parse_args()

    print(f"\n{'=' * 64}")
    print("Pharma-Z — AuditTrail Contract Deployment")
    print(f"{'=' * 64}")

    print(f"\n[1/4] Compiling {os.path.basename(SOURCE_PATH)} (solc {SOLC_VERSION})")
    abi, bytecode = compile_contract()
    with open(ABI_PATH, "w") as handle:
        json.dump(abi, handle, indent=2)
    functions = [e["name"] for e in abi if e.get("type") == "function"]
    print(f"  ABI written: {ABI_PATH}")
    print(f"  Functions: {', '.join(functions)}")
    print(f"  Bytecode: {len(bytecode) // 2} bytes")

    if args.abi_only:
        print("\n  --abi-only: stopping before deployment.\n")
        return

    print(f"\n[2/4] Connecting to {args.provider}")
    web3 = Web3(Web3.HTTPProvider(args.provider))
    if not web3.is_connected():
        print(f"  x Cannot reach {args.provider}")
        print(f"    Start a chain first:  npx ganache --port 8545 --deterministic")
        sys.exit(1)
    accounts = web3.eth.accounts
    print(f"  Connected. chain_id={web3.eth.chain_id}, accounts={len(accounts)}")
    print(f"  Deployer: {accounts[0]}")
    print(f"  Balance:  {web3.from_wei(web3.eth.get_balance(accounts[0]), 'ether')} ETH")

    print(f"\n[3/4] Deploying")
    contract = web3.eth.contract(abi=abi, bytecode=bytecode)
    tx_hash = contract.constructor().transact({"from": accounts[0]})
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    address = receipt.contractAddress
    print(f"  Address:  {address}")
    print(f"  Tx:       {receipt.transactionHash.hex()}")
    print(f"  Block:    {receipt.blockNumber}")
    print(f"  Gas used: {receipt.gasUsed:,}")

    print(f"\n[4/4] Sanity check")
    deployed = web3.eth.contract(address=address, abi=abi)
    print(f"  owner():       {deployed.functions.owner().call()}")
    print(f"  recordCount(): {deployed.functions.getRecordCount().call()}")

    if not args.no_env:
        write_env(address)
        print(f"\n  CONTRACT_ADDRESS written to {os.path.abspath(ENV_PATH)}")
    else:
        print(f"\n  CONTRACT_ADDRESS={address}   (not written, --no-env)")

    print(f"\n{'=' * 64}")
    print("  Deployed. Now run:  python scripts/test_blockchain_e2e.py")
    print(f"{'=' * 64}\n")


if __name__ == "__main__":
    main()
