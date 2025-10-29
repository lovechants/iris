import hashlib
import pathlib
import time
from typing import Optional
import Metal


class JITCache:
    def __init__(self):
        self._cache = {}
        self._cache_dir = pathlib.Path.home() / ".iris_cache"
        self._cache_dir.mkdir(exist_ok=True)

    def _key(self, src: str, fn: str) -> str:
        h = hashlib.sha1()
        h.update(src.encode("utf-8"))
        h.update(fn.encode("utf-8"))
        return h.hexdigest()

    def compile(self, source: str, function_name: str, device) -> "Metal.MTLLibrary":
        key = self._key(source, function_name)

        if key in self._cache:
            return self._cache[key]

        start = time.perf_counter()
        library, error = device.newLibraryWithSource_options_error_(source, None, None)
        end = time.perf_counter()

        if library is None:
            msg = error.localizedDescription() if error else "Metal JIT compile failed."
            raise RuntimeError(msg)

        self._cache[key] = library
        from metal_runtime.api import log_event
        log_event(function_name, (end - start) * 1000.0, "compile")

        try:
            cache_file = self._cache_dir / f"{key}.metal"
            cache_file.write_text(source)
        except Exception:
            pass

        return library


_JIT_CACHE: Optional[JITCache] = None


def get_jit_cache() -> JITCache:
    global _JIT_CACHE
    if _JIT_CACHE is None:
        _JIT_CACHE = JITCache()
    return _JIT_CACHE
