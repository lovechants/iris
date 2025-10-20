import hashlib
import subprocess
import tempfile
import Metal
import threading
from pathlib import Path
from typing import Optional, Dict 

class CompilationError(Exception):
    pass 

class MetalCompiler:
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".cache" / "iris"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._library_cache: Dict[str, "Metal.MTLLibrary"] = {}

    def _compute_hash(self, source: str) -> str:
        return hashlib.sha256(source.encode()).hexdigest()

    def _get_cached_path(self, source_hash: str) -> Path:
        return self.cache_dir / f"{source_hash}.metallib"

    def compile(self, source: str, device: "Metal.MTLDevice") -> "Metal.MTLLibrary":
        source_hash = self._compute_hash(source)

        if source_hash in self._library_cache:
            return self._library_cache[source_hash]

        cached_path = self._get_cached_path(source_hash)

        if cached_path.exists():
            # device.newLibraryWithFile_error_ returns a tuple (library, error)
            library, error = device.newLibraryWithFile_error_(str(cached_path), None)
            if library is not None:
                self._library_cache[source_hash] = library
                return library
        
        library = self._compile_source(source, device, cached_path)
        self._library_cache[source_hash] = library
        return library

    #TODO Remove the failing cache code at some point was a nice attempt to see just not supported right now
    # Will just use in memory cache.
    def _compile_source(self, source: str, device: "Metal.MTLDevice", output_path: Path) -> "Metal.MTLLibrary":
        compilation_complete = threading.Event()
        library_result = None
        error_result = None

        def completion_handler(library, error):
            nonlocal library_result, error_result
            library_result = library
            error_result = error
            compilation_complete.set() 

        device.newLibraryWithSource_options_completionHandler_(
            source,           # The MSL source code string
            None,             # MTLCompileOptions (can be None for defaults)
            completion_handler # The function to call when done
        )

        if not compilation_complete.wait(timeout=5):
            raise CompilationError("Compilation timed out after 5 seconds.")

        if error_result:
            error_description = error_result.localizedDescription()
            raise CompilationError(f"Metal Compilation Failed:\n{error_description}")
        
        if library_result is None:
            raise CompilationError("Metal Compilation Failed: Unknown error (library is None).")

        try:
            library_data = library_result.dataRepresentation()
            if library_data is None:
                print("Library.dataRepresentation() not returned None. Cannot cache to disk")
            elif not library_data:
                print("library.dataRepresentation returned empty data, cannot cache to disk")
            else:
                output_path.write_bytes(library_data)
                print(f"cached library to {output_path}")
        except Exception as e:
            print(f"Warning: Could not write compiled library to cache: {e}")

        return library_result

_compiler_instance: Optional[MetalCompiler] = None 

def get_compiler() -> MetalCompiler:
    global _compiler_instance
    if _compiler_instance is None:
        _compiler_instance = MetalCompiler()
    return _compiler_instance
