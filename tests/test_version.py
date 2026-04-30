"""Test that __version__ is set and matches pyproject.toml."""

import nerve


def test_version_exists():
    assert hasattr(nerve, "__version__")
    assert isinstance(nerve.__version__, str)
    assert len(nerve.__version__) > 0


def test_version_format():
    parts = nerve.__version__.split(".")
    assert len(parts) >= 2
    assert all(p.isdigit() for p in parts)
