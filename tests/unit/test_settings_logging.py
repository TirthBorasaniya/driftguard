"""Tests for settings, config loading, and log redaction."""

import json
import logging

import pytest

from driftguard.common.config import load_config
from driftguard.common.logging import REDACTED, configure_logging, redact_value
from driftguard.common.settings import Settings

pytestmark = pytest.mark.unit


def test_secrets_not_in_repr(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "super-secret-token")
    settings = Settings(_env_file=None)
    assert settings.admin_token.get_secret_value() == "super-secret-token"
    assert "super-secret-token" not in repr(settings)


def test_load_config_requires_version(tmp_path):
    good = tmp_path / "g.yaml"
    good.write_text("version: 2\na: 1\n")
    assert load_config(good)["a"] == 1
    bad = tmp_path / "b.yaml"
    bad.write_text("a: 1\n")
    with pytest.raises(ValueError):
        load_config(bad)


def test_redact_value_scrubs_ips():
    assert redact_value("10.1.2.3") == REDACTED
    assert redact_value("flow from 192.168.0.1 to x") == f"flow from {REDACTED} to x"
    assert redact_value("fe80::1") == REDACTED
    assert redact_value({"src_ip": "x", "n": 1}) == {"src_ip": REDACTED, "n": 1}
    assert redact_value("version 1.2.3") == "version 1.2.3"


def test_logger_never_emits_ip(capsys):
    logger = configure_logging("test", "INFO")
    logger.info("scored", src_ip="10.0.0.9", detail="dst 172.16.5.4", event_id="e1")
    logging.getLogger().handlers[0].flush()
    out = capsys.readouterr().out.strip().splitlines()[-1]
    record = json.loads(out)
    assert "10.0.0.9" not in out and "172.16.5.4" not in out
    assert record["service"] == "test" and record["event"] == "scored" and "ts" in record
