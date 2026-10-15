"""Resumable, length-checked file download with an optional MD5 check."""

import base64
import hashlib
import urllib.request
from pathlib import Path

# ============= Constants =============

CHUNK = 1 << 20
MAX_ATTEMPTS = 20


# ============= Download =============


def remote_size_and_md5(url: str, opener=urllib.request.urlopen) -> tuple[int, str | None]:
    """
    Read the content length and, when the host provides it, the hex MD5 (Google Cloud Storage).

    Returns
    -------
    size : int
        Size in bytes.
    md5_hex : str or None
        Hex MD5 from an `x-goog-hash: md5=...` header, or None.
    """
    request = urllib.request.Request(url, method="HEAD")
    with opener(request) as response:
        size = int(response.headers["Content-Length"])
        md5_hex = None
        for value in response.headers.get_all("x-goog-hash") or []:
            for part in value.split(","):
                if part.strip().startswith("md5="):
                    md5_hex = base64.b64decode(part.strip()[4:]).hex()
    return size, md5_hex


def file_md5(path: Path) -> str:
    """Hex MD5 of a file."""
    digest = hashlib.md5()  # noqa: S324 (integrity check against the host's published MD5)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(
    url: str, target: Path, max_attempts: int = MAX_ATTEMPTS, opener=urllib.request.urlopen
) -> Path:
    """
    Download `url` to `target`, resuming a `.part` file and retrying truncated transfers.

    The file is renamed into place only when its size equals the server's content length and,
    when the host publishes one, its MD5 matches.

    Parameters
    ----------
    opener : callable
        `urllib.request.urlopen` compatible callable, injectable for tests.

    Raises
    ------
    RuntimeError
        If the transfer cannot be completed or verified.
    """
    size, md5_hex = remote_size_and_md5(url, opener)
    partial = target.with_suffix(target.suffix + ".part")
    for _ in range(max_attempts):
        have = partial.stat().st_size if partial.exists() else 0
        if have == size:
            break
        if have > size:
            partial.unlink()
            have = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with opener(request) as response, open(partial, "ab") as out:
                if have and response.status != 206:
                    out.truncate(0)  # the server ignored the range request; restart
                while chunk := response.read(CHUNK):
                    out.write(chunk)
        except OSError:
            continue  # dropped connection; resume from what was written
    if not partial.exists() or partial.stat().st_size != size:
        raise RuntimeError(f"incomplete download of {url}")
    if md5_hex is not None and file_md5(partial) != md5_hex:
        partial.unlink()
        raise RuntimeError(f"MD5 mismatch for {url}")
    partial.rename(target)
    return target
