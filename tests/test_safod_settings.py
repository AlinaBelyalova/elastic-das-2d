"""Configuration imports are portable; data preparation still validates inputs."""

import importlib

import pytest

from scripts.safod import settings


def test_settings_import_does_not_scan_raw_data(monkeypatch):
    def unexpected_scan(*args, **kwargs):
        raise AssertionError("Configuration import must not scan raw data")

    monkeypatch.setattr(settings.glob, "glob", unexpected_scan)
    importlib.reload(settings)


def test_selected_files_are_required_when_preparation_starts(monkeypatch):
    monkeypatch.setattr(settings.glob, "glob", lambda pattern: [])
    with pytest.raises(FileNotFoundError, match="No files matched"):
        settings.get_selected_files()
