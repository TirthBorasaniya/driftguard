"""Regression tests for the resumable download (a truncated archive was once accepted)."""

import base64
import hashlib
import io

import pytest

from driftguard.common.download import download, file_md5

pytestmark = pytest.mark.unit
PAYLOAD = bytes(range(256)) * 4000


class FakeResponse(io.BytesIO):
    def __init__(self, body, headers, status=200):
        super().__init__(body)
        self.status = status
        self.headers = type(
            "H",
            (),
            {"get_all": lambda s, k: headers.get(k, []), "__getitem__": lambda s, k: headers[k]},
        )()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def make_opener(md5_b64, truncate_first):
    calls = {"get": 0}

    def opener(request):
        headers = {
            "Content-Length": str(len(PAYLOAD)),
            "x-goog-hash": [f"md5={md5_b64}"] if md5_b64 else [],
        }
        if request.get_method() == "HEAD":
            return FakeResponse(b"", headers)
        calls["get"] += 1
        rng = request.headers.get("Range")
        start = int(rng.split("=")[1].split("-")[0]) if rng else 0
        body = PAYLOAD[start:]
        if truncate_first and calls["get"] == 1:
            body = body[: len(body) // 2]
        return FakeResponse(body, headers, 206 if rng else 200)

    return opener, calls


def b64md5(data):
    return base64.b64encode(hashlib.md5(data).digest()).decode()  # noqa: S324


def test_truncated_transfer_is_resumed_and_verified(tmp_path):
    opener, calls = make_opener(b64md5(PAYLOAD), truncate_first=True)
    target = download("http://x/a.bin", tmp_path / "a.bin", opener=opener)
    assert target.read_bytes() == PAYLOAD and calls["get"] == 2
    assert not (tmp_path / "a.bin.part").exists()


def test_md5_mismatch_rejected_and_part_removed(tmp_path):
    opener, _ = make_opener(b64md5(b"other"), truncate_first=False)
    with pytest.raises(RuntimeError, match="MD5 mismatch"):
        download("http://x/b.bin", tmp_path / "b.bin", opener=opener)
    assert not (tmp_path / "b.bin").exists() and not (tmp_path / "b.bin.part").exists()


def test_incomplete_download_never_renamed_into_place(tmp_path):
    def opener(request):
        headers = {"Content-Length": str(len(PAYLOAD)), "x-goog-hash": []}
        body = b"" if request.get_method() == "HEAD" else PAYLOAD[:100]
        return FakeResponse(body, headers)

    with pytest.raises(RuntimeError, match="incomplete"):
        download("http://x/c.bin", tmp_path / "c.bin", max_attempts=3, opener=opener)
    assert not (tmp_path / "c.bin").exists()


def test_file_md5(tmp_path):
    path = tmp_path / "f"
    path.write_bytes(b"abc")
    assert file_md5(path) == hashlib.md5(b"abc").hexdigest()  # noqa: S324
