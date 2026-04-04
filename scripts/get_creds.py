"""
scripts/get_creds.py — Run ONCE to generate your Polymarket CLOB API credentials.

Usage:
    python scripts/get_creds.py

Paste the output into your .env file.
Your private key is used locally only — it never leaves this machine.
"""
import sys
import os

# Allow running from either the project root or the scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main() -> None:
    private_key = input("Enter your wallet private key (hex, no 0x prefix): ").strip()

    if not private_key:
        print("ERROR: private key cannot be empty.")
        sys.exit(1)

    # Strip 0x prefix if the user included it anyway
    if private_key.startswith("0x") or private_key.startswith("0X"):
        private_key = private_key[2:]

    if len(private_key) != 64:
        print(f"WARNING: expected 64 hex chars, got {len(private_key)}. Continuing anyway.")

    try:
        from py_clob_client.client import ClobClient
    except ImportError:
        print("ERROR: py_clob_client is not installed.")
        print("Run:  pip install -r requirements.txt")
        sys.exit(1)

    print("\nConnecting to Polymarket CLOB...")
    try:
        # L1 auth: only key + chain_id required (no API creds yet — we're deriving them)
        client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key,
            chain_id=137,
        )
        creds = client.create_or_derive_api_creds()
    except Exception as exc:
        print(f"ERROR: could not derive credentials — {exc}")
        print("Make sure your private key is correct and you have internet access.")
        sys.exit(1)

    print("\n✓ Success! Add these to your .env file:\n")
    print(f"POLY_PRIVATE_KEY={private_key}")
    print(f"POLY_API_KEY={creds.api_key}")
    print(f"POLY_API_SECRET={creds.api_secret}")
    print(f"POLY_API_PASSPHRASE={creds.api_passphrase}")
    print()


if __name__ == "__main__":
    main()
