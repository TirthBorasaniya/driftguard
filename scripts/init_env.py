"""Create .env from .env.example with generated secrets; never overwrites an existing file."""

import re
import secrets
import sys
from pathlib import Path

from driftguard.common.secrets import secret_problem
from driftguard.common.settings import REPO_ROOT

# ============= Constants =============

SECRET_KEYS = ("ADMIN_TOKEN", "IP_HASH_SALT", "GF_ADMIN_PASSWORD")


# ============= Generation =============


def render_env(example: str) -> str:
    """Return the example text with every empty or placeholder secret replaced by a random one."""
    out = []
    for line in example.splitlines():
        match = re.fullmatch(rf"({'|'.join(SECRET_KEYS)})=(.*)", line)
        if match and secret_problem(match.group(2)):
            line = f"{match.group(1)}={secrets.token_urlsafe(32)}"
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> int:
    """Write .env when missing; print what happened."""
    target = REPO_ROOT / ".env"
    if target.exists():
        print(".env exists; leaving it unchanged")
        return 0
    target.write_text(render_env(Path(REPO_ROOT / ".env.example").read_text()))
    target.chmod(0o600)
    print(f"Wrote {target} with generated secrets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
