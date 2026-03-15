"""
Backend registry for etops tensor operations.

This module provides a registry for backend implementations with lazy loading
and availability checking. Backends are loaded on-demand and their dependencies
are checked before use.
"""

import importlib
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from etops.config import TensorOperationConfig
    from etops.backends.base import CompiledOperation


# Registry: backend_name -> module info with factory and optimize functions
_BACKEND_REGISTRY = {
    "tpp": {
        "module": "etops.backends.tpp",
        "factory": "create_operation",
        "optimize": "optimize_config",
        "default_config": "get_default_optimization_config",
        "requires": [],  # Bundled with etops
    },
    "cutile": {
        "module": "etops.backends.cutile",
        "factory": "create_operation",
        "optimize": "optimize_config",
        "default_config": "get_default_optimization_config",
        "requires": ["cuda.tile"],
    },
}

# Cache for backend availability
_available_backends: dict[str, bool] = {}


def _check_availability(backend_name: str) -> bool:
    """
    Check if a backend's dependencies are available.
    
    Args:
        backend_name: Name of the backend to check
        
    Returns:
        True if all dependencies are available, False otherwise
    """
    if backend_name in _available_backends:
        return _available_backends[backend_name]

    if backend_name not in _BACKEND_REGISTRY:
        return False

    info = _BACKEND_REGISTRY[backend_name]
    for pkg in info["requires"]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            _available_backends[backend_name] = False
            return False

    _available_backends[backend_name] = True
    return True


def get_backend(backend_name: str) -> Callable[["TensorOperationConfig"], "CompiledOperation"]:
    """
    Get backend factory function.

    Args:
        backend_name: Name of the backend ("tpp" or "cutile")

    Returns:
        Factory function that creates compiled operations

    Raises:
        ValueError: If backend name is unknown
        ImportError: If backend dependencies are not installed
    """
    if backend_name not in _BACKEND_REGISTRY:
        available = list_backends()
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available backends: {available}"
        )

    if not _check_availability(backend_name):
        info = _BACKEND_REGISTRY[backend_name]
        raise ImportError(
            f"Backend '{backend_name}' requires: {info['requires']}. "
            f"Install with: pip install etops[{backend_name}]"
        )

    info = _BACKEND_REGISTRY[backend_name]
    module = importlib.import_module(info["module"])
    return getattr(module, info["factory"])


def list_backends() -> list[str]:
    """
    List available backends.

    Returns:
        List of backend names that have all dependencies installed.
    """
    return [name for name in _BACKEND_REGISTRY.keys()
            if _check_availability(name)]


def get_optimizer(backend_name: str):
    """
    Get backend optimizer function.

    Args:
        backend_name: Name of the backend ("tpp" or "cutile")

    Returns:
        Optimizer function that optimizes TensorOperationConfig

    Raises:
        ValueError: If backend name is unknown
        ImportError: If backend dependencies are not installed
    """
    if backend_name not in _BACKEND_REGISTRY:
        available = list_backends()
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available backends: {available}"
        )

    if not _check_availability(backend_name):
        info = _BACKEND_REGISTRY[backend_name]
        raise ImportError(
            f"Backend '{backend_name}' requires: {info['requires']}. "
            f"Install with: pip install etops[{backend_name}]"
        )

    info = _BACKEND_REGISTRY[backend_name]
    module = importlib.import_module(info["module"])
    return getattr(module, info["optimize"])


def get_default_optimization_config(backend_name: str):
    """
    Get backend's default optimization configuration.

    Args:
        backend_name: Name of the backend ("tpp" or "cutile")

    Returns:
        Default optimization configuration for the backend

    Raises:
        ValueError: If backend name is unknown
        ImportError: If backend dependencies are not installed
    """
    if backend_name not in _BACKEND_REGISTRY:
        available = list_backends()
        raise ValueError(
            f"Unknown backend '{backend_name}'. "
            f"Available backends: {available}"
        )

    if not _check_availability(backend_name):
        info = _BACKEND_REGISTRY[backend_name]
        raise ImportError(
            f"Backend '{backend_name}' requires: {info['requires']}. "
            f"Install with: pip install etops[{backend_name}]"
        )

    info = _BACKEND_REGISTRY[backend_name]
    module = importlib.import_module(info["module"])
    return getattr(module, info["default_config"])()


__all__ = ["get_backend", "get_optimizer", "get_default_optimization_config", "list_backends"]
