"""Tests for the idempotent consumer Redis dedup guard (infra improvement 5)."""

import time

import pytest

pytest.importorskip("redis")

from src.consumer.dedup import DEDUP_KEY_PREFIX, is_duplicate_event  # noqa: E402


class FakeRedis:
    """Minimal in-memory stand-in for redis-py's SET NX EX semantics."""

    def __init__(self):
        self.store = {}

    def set(self, key, value, ex=None, nx=False):
        now = time.monotonic()
        expiry = self.store.get(key)
        if expiry is not None and expiry < now:
            del self.store[key]

        if nx and key in self.store:
            return None

        self.store[key] = now + ex if ex else float("inf")
        return True


def test_first_call_is_not_duplicate():
    redis_client = FakeRedis()
    assert is_duplicate_event(redis_client, "event-1") is False


def test_second_call_within_ttl_is_duplicate():
    redis_client = FakeRedis()
    is_duplicate_event(redis_client, "event-1")
    assert is_duplicate_event(redis_client, "event-1") is True


def test_different_events_are_independent():
    redis_client = FakeRedis()
    assert is_duplicate_event(redis_client, "event-1") is False
    assert is_duplicate_event(redis_client, "event-2") is False


def test_key_uses_dedup_prefix():
    redis_client = FakeRedis()
    is_duplicate_event(redis_client, "event-1")
    assert f"{DEDUP_KEY_PREFIX}event-1" in redis_client.store
