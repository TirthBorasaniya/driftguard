"""Restore the previous champion through the token-protected admin API (`make rollback`)."""

import argparse
import sys

import httpx

from driftguard.common.settings import get_settings

# ============= CLI =============


def main() -> int:
    """Call POST /admin/rollback and print the logged alias change."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--reason", default="operator rollback")
    args = parser.parse_args()
    settings = get_settings()
    response = httpx.post(
        f"{settings.serving_url}/admin/rollback",
        json={"dataset": args.dataset, "reason": args.reason},
        headers={"Authorization": f"Bearer {settings.admin_token.get_secret_value()}"},
        timeout=30,
    )
    if response.status_code != 200:
        print(f"Rollback failed ({response.status_code}): {response.text}")
        return 1
    entry = response.json()
    print(f"Champion is now version {entry['champion']} (was {entry['rolled_back_from']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
