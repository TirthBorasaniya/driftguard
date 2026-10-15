"""Structured JSON logging with IP redaction for every DriftGuard service."""

import ipaddress
import logging
import re
import sys
from typing import Any

import structlog

# ============= Constants =============

REDACTED = "[redacted]"
SENSITIVE_KEYS = frozenset({"src_ip", "dst_ip", "source_ip", "destination_ip", "ip", "salt"})
IPV4_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")


# ============= Redaction =============


def _looks_like_ip(token: str) -> bool:
    """Return True if a token parses as an IPv4 or IPv6 address."""
    try:
        ipaddress.ip_address(token)
    except ValueError:
        return False
    return True


def redact_value(value: Any) -> Any:
    """
    Remove IP addresses from a log value.

    Parameters
    ----------
    value : Any
        Value of a log field. Strings are scanned for IPv4 literals and whole-string IPv6
        literals; containers are processed recursively.

    Returns
    -------
    redacted : Any
        Value with IP addresses replaced by a placeholder.
    """
    if isinstance(value, str):
        if _looks_like_ip(value.strip()):
            return REDACTED
        return IPV4_PATTERN.sub(
            lambda match: REDACTED if _looks_like_ip(match.group(0)) else match.group(0), value
        )
    if isinstance(value, dict):
        return {
            key: REDACTED if key in SENSITIVE_KEYS else redact_value(v) for key, v in value.items()
        }
    if isinstance(value, list | tuple):
        return type(value)(redact_value(v) for v in value)
    return value


def redact_processor(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Structlog processor that drops sensitive keys and scrubs IP literals."""
    return {
        key: REDACTED if key in SENSITIVE_KEYS else redact_value(value)
        for key, value in event_dict.items()
    }


# ============= Configuration =============


def configure_logging(service: str, level: str = "INFO") -> structlog.stdlib.BoundLogger:
    """
    Configure structlog to emit one JSON object per line on stdout.

    Parameters
    ----------
    service : str
        Service name added to every record as `service`.
    level : str
        Minimum log level name.

    Returns
    -------
    logger : structlog.stdlib.BoundLogger
        Logger bound to the service name.
    """
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper(), force=True)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True, key="ts"),
            structlog.processors.format_exc_info,
            redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    return structlog.get_logger().bind(service=service)


def get_logger(service: str = "driftguard") -> structlog.stdlib.BoundLogger:
    """Return a logger bound to a service name without reconfiguring structlog."""
    return structlog.get_logger().bind(service=service)
