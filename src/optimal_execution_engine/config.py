"""Typed configuration loading for the execution engine."""

import os
from pathlib import Path
import tomllib

from dotenv import load_dotenv
from pydantic import BaseModel


class CacheSettings(BaseModel):
    """Filesystem location and validation settings for cached market data."""

    root_dir: str


class ClickHouseSettings(BaseModel):
    """Environment-backed ClickHouse connection fields."""

    host: str | None = None
    port: int | None = None
    user: str | None = None
    password: str | None = None
    secure: bool = False
    verify: bool = False


class Settings(BaseModel):
    """Top-level runtime settings."""

    cache: CacheSettings
    clickhouse: ClickHouseSettings = ClickHouseSettings()


def _parse_bool(value: str | bool | None, default: bool) -> bool:
    """Parse permissive boolean input from env or config values."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _read_optional_env_var(env_name: str) -> str | None:
    """Read one environment variable and normalize empty values to ``None``.

    Parameters
    ----------
    env_name
        Name of the environment variable to read.

    Returns
    -------
    str | None
        Non-empty environment value when present, otherwise ``None``.
    """
    env_value = os.getenv(env_name)
    if env_value is None:
        return None

    normalized_value = env_value.strip()
    if normalized_value == "":
        return None

    return normalized_value


def _parse_optional_port(value: str | int | None) -> int | None:
    """Convert an optional configured port value into an ``int`` or ``None``.

    Parameters
    ----------
    value
        Port candidate from environment or TOML config.

    Returns
    -------
    int | None
        Parsed port value. Zero and empty inputs map to ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None

    parsed_port = int(value)
    if parsed_port == 0:
        return None

    return parsed_port


def _resolve_clickhouse_settings(
    raw_clickhouse: dict[str, object],
) -> ClickHouseSettings:
    """Merge TOML ClickHouse fields with environment overrides.

    Parameters
    ----------
    raw_clickhouse
        ClickHouse subsection from ``config.toml``.

    Returns
    -------
    ClickHouseSettings
        Normalized ClickHouse settings with environment overrides applied.
    """
    # Environment values win when present so local secrets override tracked config.
    env_host = _read_optional_env_var("CLICKHOUSE_HOST")
    env_port = _read_optional_env_var("CLICKHOUSE_PORT")
    env_user = _read_optional_env_var("CLICKHOUSE_USER")
    env_password = _read_optional_env_var("CLICKHOUSE_PASSWORD")
    env_secure = _read_optional_env_var("CLICKHOUSE_SECURE")
    env_verify = _read_optional_env_var("CLICKHOUSE_VERIFY")

    resolved_host = env_host or raw_clickhouse.get("host")
    resolved_port_source = (
        env_port if env_port is not None else raw_clickhouse.get("port")
    )
    resolved_user = env_user or raw_clickhouse.get("user")
    resolved_password = env_password or raw_clickhouse.get("password")

    # Parse booleans through the same permissive parser for env and TOML inputs.
    resolved_secure = _parse_bool(
        env_secure,
        _parse_bool(raw_clickhouse.get("secure"), False),
    )
    resolved_verify = _parse_bool(
        env_verify,
        _parse_bool(raw_clickhouse.get("verify"), False),
    )

    return ClickHouseSettings(
        host=resolved_host if resolved_host else None,
        port=_parse_optional_port(resolved_port_source),
        user=resolved_user if resolved_user else None,
        password=resolved_password if resolved_password else None,
        secure=resolved_secure,
        verify=resolved_verify,
    )


def load_settings(config_path: Path) -> Settings:
    """Read TOML settings from disk and apply environment overrides.

    Parameters
    ----------
    config_path
        Path to the project ``config.toml`` file.

    Returns
    -------
    Settings
        Validated runtime settings for cache and optional ClickHouse access.
    """
    load_dotenv()

    # Read the file once, then normalize nested sections before validation.
    raw_settings = tomllib.loads(config_path.read_text(encoding="utf-8"))

    raw_clickhouse = dict(raw_settings.get("clickhouse", {}))
    raw_settings["clickhouse"] = _resolve_clickhouse_settings(raw_clickhouse)
    return Settings.model_validate(raw_settings)
