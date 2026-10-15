"""Salted, irreversible keys for source and destination hosts; raw IPs never leave the consumer."""

import hashlib

from driftguard.common.secrets import require_secret

# ============= Keys =============


class HostKeyer:
    """
    Derives `sha256(salt + ip)` keys.

    Parameters
    ----------
    salt : str
        Secret salt from the environment (`IP_HASH_SALT`); rejected when empty, a placeholder, or
        shorter than 16 characters, because a known salt makes the keys reversible by enumerating
        IPv4 space.
    """

    def __init__(self, salt: str):
        require_secret("IP_HASH_SALT", salt)
        self._salt = salt.encode()
        self._cache: dict[str, str] = {}

    def key(self, ip: str) -> str:
        """Return the hex SHA-256 key of an IP (cached per process; the cache never leaves it)."""
        cached = self._cache.get(ip)
        if cached is None:
            cached = hashlib.sha256(self._salt + ip.encode()).hexdigest()
            if len(self._cache) > 1_000_000:
                self._cache.clear()
            self._cache[ip] = cached
        return cached
