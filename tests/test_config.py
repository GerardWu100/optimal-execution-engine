"""Configuration and packaging tests for runtime settings."""

from pathlib import Path

from optimal_execution_engine.config import load_settings
from optimal_execution_engine.version import __version__


def test_version_is_exposed() -> None:
    """Package import should expose a version string."""
    assert isinstance(__version__, str)
    assert __version__


def test_load_settings_reads_cache_directory(tmp_path: Path) -> None:
    """The loader should return the configured cache directory."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[cache]\nroot_dir = "data/raw"\n', encoding="utf-8")

    settings = load_settings(config_path=config_path)

    assert settings.cache.root_dir == "data/raw"


def test_load_settings_applies_clickhouse_env_override(
    tmp_path: Path, monkeypatch
) -> None:
    """Environment variables should override ClickHouse defaults."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[cache]\nroot_dir = "data/raw"\n', encoding="utf-8")
    monkeypatch.setenv("CLICKHOUSE_HOST", "127.0.0.1")

    settings = load_settings(config_path=config_path)

    assert settings.clickhouse.host == "127.0.0.1"
