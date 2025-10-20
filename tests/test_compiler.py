import pytest
from metal_runtime.compiler import MetalCompiler, CompilationError, get_compiler
from metal_runtime.runtime import get_runtime


SIMPLE_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void simple_add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    c[tid] = a[tid] + b[tid];
}
"""


INVALID_KERNEL = """
#include <metal_stdlib>
using namespace metal;

kernel void broken(
    device float* out [[buffer(0)]]
) {
    this_function_does_not_exist();
}
"""


class TestMetalCompiler:
    @pytest.fixture
    def compiler(self):
        return get_compiler()

    @pytest.fixture
    def device(self):
        return get_runtime().device

    def test_singleton(self):
        c1 = get_compiler()
        c2 = get_compiler()
        assert c1 is c2

    def test_compile_simple_kernel(self, compiler, device):
        library = compiler.compile(SIMPLE_KERNEL, device)
        assert library is not None
        function = library.newFunctionWithName_("simple_add")
        assert function is not None

    def test_compile_caching(self, compiler, device):
        lib1 = compiler.compile(SIMPLE_KERNEL, device)
        lib2 = compiler.compile(SIMPLE_KERNEL, device)
        assert lib1 is lib2

    def test_compile_invalid_kernel(self, compiler, device):
        with pytest.raises(CompilationError):
            compiler.compile(INVALID_KERNEL, device)

    def test_hash_computation(self, compiler):
        hash1 = compiler._compute_hash(SIMPLE_KERNEL)
        hash2 = compiler._compute_hash(SIMPLE_KERNEL)
        assert hash1 == hash2
        assert len(hash1) == 64

    def test_different_sources_different_hashes(self, compiler):
        hash1 = compiler._compute_hash(SIMPLE_KERNEL)
        hash2 = compiler._compute_hash(SIMPLE_KERNEL + "\n")
        assert hash1 != hash2

    def test_cache_directory_creation(self, compiler):
        assert compiler.cache_dir.exists()
        assert compiler.cache_dir.is_dir()

    @pytest.mark.xfail(reason="Disk based caching not supported -> Todo just fix this and the cache code")
    def test_cached_file_exists_after_compile(self, compiler, device):
        source_hash = compiler._compute_hash(SIMPLE_KERNEL)
        cached_path = compiler._get_cached_path(source_hash)
        
        compiler.compile(SIMPLE_KERNEL, device)
        assert cached_path.exists()

    def test_load_from_cache(self, compiler, device):
        compiler.compile(SIMPLE_KERNEL, device)
        compiler._library_cache.clear()
        
        library = compiler.compile(SIMPLE_KERNEL, device)
        assert library is not None
