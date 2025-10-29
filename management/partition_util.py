import math
from typing import Tuple, Dict, Any, Optional
import Metal  # PyObjC for Metal
from metal_runtime import api


class GPUPartitioner:
    """
    Utility to help determine optimal grid and block sizes for GPU execution.

    This class provides methods to calculate appropriate grid and block dimensions
    based on the workload size and actual GPU capabilities.
    """

    def __init__(self):
        self._gpu_properties = None
        self._device = None

    @property
    def device(self):
        """Get the current Metal device."""
        if self._device is None:
            self._device = Metal.MTLCreateSystemDefaultDevice()
        return self._device

    @property
    def gpu_properties(self) -> Dict[str, Any]:
        """Get the actual properties of the current GPU device."""
        if self._gpu_properties is None:
            self._gpu_properties = self._query_gpu_properties()
        return self._gpu_properties

    def _query_gpu_properties(self) -> Dict[str, Any]:
        """Query the GPU for its actual properties using Metal API."""
        device = self.device

        properties = {
            "name": device.name(),
        }

        try:
            max_threads_size = device.maxThreadsPerThreadgroup()
            properties["max_threads_per_threadgroup"] = max_threads_size.width
        except AttributeError:
            properties["max_threads_per_threadgroup"] = 1024

        try:
            properties["thread_execution_width"] = device.threadExecutionWidth()
        except AttributeError:
            properties["thread_execution_width"] = 32

        try:
            max_total_threads_size = device.maxTotalThreadsPerThreadgroup()
            properties["max_total_threads_per_threadgroup"] = (
                max_total_threads_size.width
            )
        except AttributeError:
            properties["max_total_threads_per_threadgroup"] = properties[
                "max_threads_per_threadgroup"
            ]

        try:
            properties["threadgroup_memory_size"] = device.maxThreadgroupMemoryLength()
        except AttributeError:
            properties["threadgroup_memory_size"] = 32768

        try:
            properties["recommended_max_working_set_size"] = (
                device.recommendedMaxWorkingSetSize()
            )
        except AttributeError:
            properties["recommended_max_working_set_size"] = 268435456  # 256MB default

        try:
            properties["has_unified_memory"] = device.hasUnifiedMemory()
        except AttributeError:
            properties["has_unified_memory"] = True

        properties["max_threadgroups_per_grid"] = (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF)

        try:
            properties["supports_family"] = str(device.featureSetFamily())
        except AttributeError:
            properties["supports_family"] = "Unknown"

        try:
            properties["is_low_power"] = device.isLowPower()
        except AttributeError:
            properties["is_low_power"] = False

        try:
            properties["is_headless"] = device.isHeadless()
        except AttributeError:
            properties["is_headless"] = False

        try:
            properties["is_removable"] = device.isRemovable()
        except AttributeError:
            properties["is_removable"] = False

        try:
            if hasattr(device, "gpuFamily"):
                properties["family"] = str(device.gpuFamily())
            elif hasattr(device, "family"):
                properties["family"] = str(device.family())
            else:
                properties["family"] = "Unknown"
        except AttributeError:
            properties["family"] = "Unknown"

        if hasattr(device, "maxBufferLength"):
            properties["max_buffer_length"] = device.maxBufferLength()

        if hasattr(device, "registryID"):
            properties["registry_id"] = device.registryID()

        try:
            if hasattr(device, "architecture"):
                properties["architecture"] = str(device.architecture())
        except AttributeError:
            properties["architecture"] = "Unknown"

        return properties

    def recommend_block_size(
        self,
        threads_needed: int,
        memory_per_thread: int = 0,
        prefer_power_of_two: bool = True,
    ) -> int:
        """
        Recommend an optimal block size (threads per threadgroup) for a given workload.

        Args:
            threads_needed: Total number of threads needed for the computation
            memory_per_thread: Memory needed per thread in bytes (for shared memory calculations)
            prefer_power_of_two: Whether to prefer power-of-two block sizes (often more efficient)

        Returns:
            Recommended block size (threads per threadgroup)
        """
        max_threads = self.gpu_properties["max_threads_per_threadgroup"]
        thread_execution_width = self.gpu_properties["thread_execution_width"]

        recommended_block_size = min(threads_needed, max_threads)

        if memory_per_thread > 0:
            max_memory_threads = (
                self.gpu_properties["threadgroup_memory_size"] // memory_per_thread
            )
            recommended_block_size = min(recommended_block_size, max_memory_threads)

        # For Apple Silicon, it's often optimal to use multiples of thread_execution_width
        # But we should also consider the specific GPU family
        if prefer_power_of_two:
            recommended_block_size = 2 ** (recommended_block_size.bit_length() - 1)

            if recommended_block_size > thread_execution_width:
                recommended_block_size = (
                    recommended_block_size // thread_execution_width
                ) * thread_execution_width
            elif (
                recommended_block_size < thread_execution_width
                and threads_needed >= thread_execution_width
            ):
                recommended_block_size = thread_execution_width

        return recommended_block_size

    def recommend_grid_size(
        self,
        total_elements: int,
        block_size: int,
        max_grid_dimension: Optional[int] = None,
    ) -> int:
        """
        Recommend an optimal grid size (threadgroups) for a given workload.

        Args:
            total_elements: Total number of elements to process
            block_size: Number of threads per threadgroup
            max_grid_dimension: Maximum grid dimension (if None, use GPU default)

        Returns:
            Recommended grid size (threadgroups)
        """
        if max_grid_dimension is None:
            # For Apple Silicon, we don't have explicit grid size limits
            # but we should consider practical limits based on memory and performance
            max_grid_dimension = 65535

        # Calculate minimum number of threadgroups needed
        min_grid_size = math.ceil(total_elements / block_size)

        return min(min_grid_size, max_grid_dimension)

    def partition_1d(
        self,
        data_size: int,
        memory_per_thread: int = 0,
        max_grid_size: Optional[int] = None,
    ) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Partition a 1D workload into optimal grid and block sizes.

        Args:
            data_size: Total number of elements to process
            memory_per_thread: Memory needed per thread in bytes
            max_grid_size: Maximum grid size to use

        Returns:
            Tuple of (grid_size, block_size) for 1D partitioning
        """
        block_size = self.recommend_block_size(data_size, memory_per_thread)
        grid_size = self.recommend_grid_size(data_size, block_size, max_grid_size)

        return (grid_size,), (block_size,)

    def partition_2d(
        self, width: int, height: int, memory_per_thread: int = 0
    ) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """
        Partition a 2D workload into optimal grid and block sizes.

        Args:
            width: Width of the 2D data
            height: Height of the 2D data
            memory_per_thread: Memory needed per thread in bytes

        Returns:
            Tuple of (grid_size, block_size) for 2D partitioning
        """
        # For 2D, we need to consider both dimensions
        max_threads = self.gpu_properties["max_threads_per_threadgroup"]
        thread_execution_width = self.gpu_properties["thread_execution_width"]

        block_dim = int(math.sqrt(max_threads))

        block_dim = (block_dim // thread_execution_width) * thread_execution_width
        if block_dim == 0:
            block_dim = thread_execution_width

        while block_dim * block_dim > max_threads:
            block_dim -= thread_execution_width

        block_size = (block_dim, block_dim)

        grid_x = math.ceil(width / block_size[0])
        grid_y = math.ceil(height / block_size[1])

        grid_size = (grid_x, grid_y)

        return grid_size, block_size

    def explain_partition(
        self, data_size: int, grid_size: Tuple[int, ...], block_size: Tuple[int, ...]
    ) -> str:
        """
        Generate a human-readable explanation of the partitioning.

        Args:
            data_size: Total number of elements to process
            grid_size: Grid dimensions
            block_size: Block dimensions

        Returns:
            A string explaining the partitioning
        """
        total_threads = 1
        for dim in grid_size:
            total_threads *= dim
        for dim in block_size:
            total_threads *= dim

        utilization = min(100.0, (data_size / total_threads) * 100)

        explanation = (
            f"For {data_size} elements:\n"
            f"  Grid size: {grid_size} (total threadgroups: {math.prod(grid_size)})\n"
            f"  Block size: {block_size} (threads per threadgroup: {math.prod(block_size)})\n"
            f"  Total threads: {total_threads}\n"
            f"  Utilization: {utilization:.1f}%\n"
            f"  GPU: {self.gpu_properties['name']}\n"
            f"  GPU family: {self.gpu_properties['family']}\n"
            f"  Thread execution width: {self.gpu_properties['thread_execution_width']}\n"
            f"  Max threads per threadgroup: {self.gpu_properties['max_threads_per_threadgroup']}\n"
            f"  Threadgroup memory size: {self.gpu_properties['threadgroup_memory_size']} bytes\n"
        )

        return explanation

    def print_device_info(self):
        """Print detailed information about the GPU device."""
        props = self.gpu_properties
        print("GPU Device Information:")
        print("=" * 50)
        print(f"Name: {props['name']}")
        print(f"Family: {props['family']}")
        print(f"Thread Execution Width: {props['thread_execution_width']}")
        print(f"Max Threads per Threadgroup: {props['max_threads_per_threadgroup']}")
        print(
            f"Max Total Threads per Threadgroup: {props['max_total_threads_per_threadgroup']}"
        )
        print(f"Threadgroup Memory Size: {props['threadgroup_memory_size']} bytes")
        print(
            f"Recommended Max Working Set Size: {props['recommended_max_working_set_size']} bytes"
        )
        print(f"Has Unified Memory: {props['has_unified_memory']}")
        print(f"Supports Family: {props['supports_family']}")
        print(f"Is Low Power: {props['is_low_power']}")
        print(f"Is Headless: {props['is_headless']}")
        print(f"Is Removable: {props['is_removable']}")

        if "max_buffer_length" in props:
            print(f"Max Buffer Length: {props['max_buffer_length']} bytes")
        if "registry_id" in props:
            print(f"Registry ID: {props['registry_id']}")


partitioner = GPUPartitioner()


def partition_1d(
    data_size: int, memory_per_thread: int = 0, max_grid_size: Optional[int] = None
) -> Tuple[Tuple[int], Tuple[int]]:
    """
    Partition a 1D workload into optimal grid and block sizes.

    Args:
        data_size: Total number of elements to process
        memory_per_thread: Memory needed per thread in bytes
        max_grid_size: Maximum grid size to use

    Returns:
        Tuple of (grid_size, block_size) for 1D partitioning
    """
    return partitioner.partition_1d(data_size, memory_per_thread, max_grid_size)


def partition_2d(
    width: int, height: int, memory_per_thread: int = 0
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Partition a 2D workload into optimal grid and block sizes.

    Args:
        width: Width of the 2D data
        height: Height of the 2D data
        memory_per_thread: Memory needed per thread in bytes

    Returns:
        Tuple of (grid_size, block_size) for 2D partitioning
    """
    return partitioner.partition_2d(width, height, memory_per_thread)


def explain_partition(
    data_size: int, grid_size: Tuple[int, ...], block_size: Tuple[int, ...]
) -> str:
    """
    Generate a human-readable explanation of the partitioning.

    Args:
        data_size: Total number of elements to process
        grid_size: Grid dimensions
        block_size: Block dimensions

    Returns:
        A string explaining the partitioning
    """
    return partitioner.explain_partition(data_size, grid_size, block_size)


def print_device_info():
    """Print detailed information about the GPU device."""
    partitioner.print_device_info()
