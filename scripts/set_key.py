"""Store the API key in the OS credential store instead of a file.

    python set_key.py            # prompts, stores
    python set_key.py --show     # report whether a key is stored (never prints it)
    python set_key.py --delete   # remove it

The key is read via getpass, so it is not echoed to the terminal and does not
land in shell history. On Windows it goes to Credential Manager (DPAPI, tied to
your Windows account); on macOS the Keychain; on Linux whatever Secret Service
backend is available.

This protects the key at rest — it is not sitting in a file you might commit,
zip up, or reveal on a screen-share. It does not protect against malware
running as you, which can read the credential store the same way this does.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import keyring
except ImportError:
    print("keyring is not installed. Run: pip install keyring", file=sys.stderr)
    raise SystemExit(1)

from bioage.recommendations.service import KEYRING_SERVICE, KEYRING_USERNAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--show", action="store_true", help="Is a key stored?")
    group.add_argument("--delete", action="store_true", help="Remove the stored key.")
    args = parser.parse_args()

    backend = keyring.get_keyring().__class__.__name__

    if args.show:
        stored = keyring.get_password(KEYRING_SERVICE, KEYRING_USERNAME)
        if stored:
            # Show only enough to identify which key, never the key.
            print(f"A key is stored in {backend} (…{stored[-4:]}).")
        else:
            print(f"No key stored in {backend}.")
        return 0

    if args.delete:
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_USERNAME)
        except keyring.errors.PasswordDeleteError:
            print("No key was stored.")
            return 0
        print("Key deleted.")
        return 0

    key = getpass.getpass("API key (input hidden): ").strip()
    if not key:
        print("Nothing entered; aborted.", file=sys.stderr)
        return 1

    keyring.set_password(KEYRING_SERVICE, KEYRING_USERNAME, key)
    print(f"Stored in {backend}. You can now delete RECOMMENDATIONS_API_KEY from .env.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
