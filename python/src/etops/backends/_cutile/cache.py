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
    
    Example:
        >>> cache = InMemoryCache()
        >>> module = cache.get_or_compile(config_parser, jit_compiler)
        >>> # Subsequent calls with same config return cached module
        >>> module2 = cache.get_or_compile(config_parser, jit_compiler)
        >>> assert module is module2
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
        hasher.update(str(config_parser.grid_size).encode())
        hasher.update(str(config_parser.dim_sizes).encode())
        hasher.update(str(config_parser.kernel_shape_m).encode())
        hasher.update(str(config_parser.kernel_shape_n).encode())
        hasher.update(str(config_parser.kernel_shape_k).encode())
        hasher.update(str(config_parser.data_type).encode())
        hasher.update(str(config_parser.exec_types).encode())
        hasher.update(str(config_parser.seq_loop_ids).encode())
        hasher.update(str(config_parser.shared_loop_ids).encode())
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
    
    def import_from_file(self, path, metadata: Dict[str, Any]) -> types.ModuleType:
        """
        Import kernel from file and cache it.
        
        Args:
            path: Directory containing kernel.py and metadata.json. Can be str or Path.
            metadata: Metadata dictionary with cache_key, grid_size, etc.
            
        Returns:
            Imported and cached kernel module
            
        Raises:
            FileNotFoundError: If kernel.py doesn't exist
        """
        path = Path(path)
        key = metadata.get("cache_key") or self._compute_key_from_metadata(metadata)
        
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            
            kernel_path = path / "kernel.py"
            if not kernel_path.exists():
                raise FileNotFoundError(f"Kernel file not found: {kernel_path}")
            
            with open(kernel_path, 'r') as f:
                kernel_code = f.read()
            
            # Use fake filename for linecache injection
            fake_filename = f"<cutile_imported_{key}>"
            module = types.ModuleType(f"cutile_kernel_{key}")
            module.__file__ = fake_filename
            module.GRID_SIZE = metadata.get("grid_size", 1)
            
            # Inject source into linecache for inspect.getsourcelines()
            source_lines = kernel_code.splitlines(keepends=True)
            linecache.cache[fake_filename] = (
                len(kernel_code),
                None,
                source_lines,
                fake_filename
            )
            
            code_obj = compile(kernel_code, fake_filename, 'exec')
            exec(code_obj, module.__dict__)
            
            self._cache[key] = module
            return module
    
    def export_to_file(self, path, key: str, kernel_code: str,
                       config_parser, einsum_str: str) -> None:
        """
        Export kernel and metadata to directory.
        
        Creates a directory with kernel.py and metadata.json files
        that can be imported later using import_from_file().
        
        Args:
            path: Directory to export to (will be created if needed). Can be str or Path.
            key: Cache key for the kernel
            kernel_code: Generated kernel Python code
            config_parser: ConfigParser with kernel configuration
            einsum_str: Original einsum string
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Write kernel code
        with open(path / "kernel.py", 'w') as f:
            f.write(kernel_code)
        
        # Write metadata
        metadata = {
            "cache_key": key,
            "grid_size": config_parser.grid_size,
            "dim_sizes": config_parser.dim_sizes,
            "kernel_shape": {
                "m": config_parser.kernel_shape_m,
                "n": config_parser.kernel_shape_n,
                "k": config_parser.kernel_shape_k,
            },
            "data_type": str(config_parser.data_type),
            "einsum_str": einsum_str,
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _compute_key_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Compute cache key from metadata dictionary.
        
        Used when importing kernels that don't have a cache_key in metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            16-character hex string cache key
        """
        hasher = hashlib.sha256()
        hasher.update(str(metadata.get("grid_size", "")).encode())
        hasher.update(str(metadata.get("dim_sizes", {})).encode())
        kernel_shape = metadata.get("kernel_shape", {})
        hasher.update(str(kernel_shape.get("m", "")).encode())
        hasher.update(str(kernel_shape.get("n", "")).encode())
        hasher.update(str(kernel_shape.get("k", "")).encode())
        hasher.update(str(metadata.get("data_type", "")).encode())
        return hasher.hexdigest()[:16]
    
    def clear(self) -> None:
        """Clear all cached kernels."""
        with self._lock:
            self._cache.clear()
    
    def __len__(self) -> int:
        """Return number of cached kernels."""
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        """Check if a key is in the cache."""
        with self._lock:
            return key in self._cache
