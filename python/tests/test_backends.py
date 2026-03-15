"""
Tests for etops backend registry.
"""

import pytest
from etops.backends import get_backend, list_backends


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_list_backends_includes_tpp(self):
        """Test that TPP backend is always available."""
        backends = list_backends()
        assert "tpp" in backends

    def test_get_backend_tpp(self):
        """Test getting TPP backend factory."""
        factory = get_backend("tpp")
        assert callable(factory)

    def test_get_backend_unknown(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("unknown_backend")