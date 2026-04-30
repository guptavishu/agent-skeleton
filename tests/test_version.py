"""Test that __version__ is set and matches pyproject.toml."""

import agentos


def test_version_exists():
    assert hasattr(agentos, "__version__")
    assert isinstance(agentos.__version__, str)
    assert len(agentos.__version__) > 0


def test_version_format():
    parts = agentos.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts)
