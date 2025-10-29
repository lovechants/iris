from typing import Dict, Any, Callable
from functools import wraps
from metal_runtime.codegen import generate_msl
from metal_runtime.api import register_kernel


class KernelFunction:
    def __init__(self, func: Callable, param_types: Dict[str, str], function_name: str):
        self.func = func
        self.param_types = param_types
        self.function_name = function_name
        self._msl_source = None

    @property
    def msl_source(self) -> str:
        if self._msl_source is None:
            self._msl_source = generate_msl(
                self.func, self.function_name, self.param_types
            )
        return self._msl_source

    def __call__(self, *args, **kwargs):
        raise RuntimeError(
            "Kernel functions cannot be called directly."
            "Use launch_kernel() or launch() to execute on GPU"
        )


def kernel(func: Callable = None, **kwargs) -> KernelFunction:
    param_types = kwargs.get("param_types", {})
    function_name = kwargs.get("name", func.__name__ if func else None)

    def decorator(f: Callable) -> KernelFunction:
        nonlocal function_name
        if function_name is None:
            function_name = f.__name__

        kernel_func = KernelFunction(f, param_types, function_name)
        register_kernel(function_name, kernel_func.msl_source, function_name)
        return kernel_func

    if func is None:
        return decorator
    else:
        return decorator(func)


class metal:
    @staticmethod
    def thread_id() -> int:
        pass

    @staticmethod
    def program_id(axis: int = 0) -> int:
        pass

    @staticmethod
    def threadgroup_id(axis: int = 0) -> int:
        pass


    @staticmethod
    def thread_id_x() -> int:
        pass

    @staticmethod
    def thread_id_y() -> int:
        pass
    
    @staticmethod
    def thread_id_2d() -> tuple[int, int]:
        pass
    
    @staticmethod
    def thread_id_z() -> int:
        pass
    
    @staticmethod
    def thread_id_3d() -> tuple[int, int, int]:
        pass
