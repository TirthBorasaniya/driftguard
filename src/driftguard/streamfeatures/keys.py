"""Salted, irreversible keys for source and destination hosts; raw IPs never leave the consumer."""

import hashlib

# ============= Keys =============


class HostKeyer:
    """
    Derives `sha256(salt + ip)` keys.

    Parameters
    ----------
    salt : str
        Secret salt from the environment (`IP_HASH_SALT`); required.
    """

    def __init__(self, salt: str):
        if not salt:
            raise ValueError("IP_HASH_SALT must be set to compute stream-feature keys")
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
