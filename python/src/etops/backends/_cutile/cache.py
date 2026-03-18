"""
In-memory kernel cache for compiled CuTile kernels.

Provides thread-safe caching of compiled kernels without file system dependencies.
Uses linecache injection to make source code available to inspect.getsourcelines().
"""

import hashlib
import linecache
import threading
import types
from pathlib import Path
from typing import Dict, Optional, Any
import json


class InMemoryCache:
    """
    Thread-safe in-memory cache for compiled kernels.
    
    Kernels are cached by configuration hash and persist until
    the cache is destroyed or explicitly cleared.
    """
    
    def __init__(self):
        self._cache: Dict[str, types.ModuleType] = {}
        self._lock = threading.RLock()
    
    def compute_key(self, config_parser) -> str:
        """
        Generate unique cache key from configuration.
        
        Args:
            config_parser: ConfigParser containing kernel configuration
            
        Returns:
            16-character hex string cache key
        """
        hasher = hashlib.sha256()
        hasher.update(config_parser.config.to_json().encode())
        return hasher.hexdigest()[:16]
    
    def get(self, key: str) -> Optional[types.ModuleType]:
        """
        Get cached module by key.
        
        Args:
            key: Cache key from compute_key()
            
        Returns:
            Cached module or None if not found
        """
        with self._lock:
            return self._cache.get(key)
    
    def put(self, key: str, module: types.ModuleType) -> None:
        """
        Store module in cache.
        
        Args:
            key: Cache key
            module: Compiled kernel module
        """
        with self._lock:
            self._cache[key] = module
    
    def get_or_compile(self, config_parser, jit_compiler) -> types.ModuleType:
        """
        Get cached kernel or compile and cache it.
        
        This method is thread-safe. If multiple threads attempt to compile
        the same kernel simultaneously, only one will perform the compilation.
        
        Args:
            config_parser: ConfigParser containing kernel configuration
            jit_compiler: JitCompiler instance to generate kernel code
            
        Returns:
            Compiled kernel module
        """
        key = self.compute_key(config_parser)
        
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            
            # Compile new kernel
            jit_compiler.jit_kernel()
            kernel_source = jit_compiler.string_kernel
            
            # Create module in-memory with unique fake filename
            fake_filename = f"<cutile_generated_{key}>"
            module = types.ModuleType(f"cutile_kernel_{key}")
            module.__file__ = fake_filename
            module.GRID_SIZE = config_parser.grid_size
            
            # Inject source into linecache so inspect.getsourcelines() can find it
            # This is required for cuda.tile's AST-to-HIR lowering which uses inspect
            source_lines = kernel_source.splitlines(keepends=True)
            # linecache.cache format: (size, mtime, lines, fullname)
            # Use None for mtime to indicate in-memory source
            linecache.cache[fake_filename] = (
                len(kernel_source),
                None,  # mtime=None indicates in-memory
                source_lines,
                fake_filename
            )
            
            code_obj = compile(kernel_source, fake_filename, 'exec')
            exec(code_obj, module.__dict__)
            
            self._cache[key] = module
            return module