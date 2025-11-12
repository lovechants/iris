import json
import time
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager

"""
Better to make this work with python native logging -> looks more professional and standard as well
Instead of writing it from scratch, this is my first time working with this so a few comments to learn
"""

# Define a custom log level for profiling
# This value is between INFO (20) and WARNING (30)
PROFILE_LEVEL = 25
logging.addLevelName(PROFILE_LEVEL, "PROFILE")


class LogLevel(Enum):
    # Align enum values with standard Python logging levels for clarity
    DEBUG = logging.DEBUG  # 10
    INFO = logging.INFO  # 20
    PROFILE = PROFILE_LEVEL  # 25
    WARNING = logging.WARNING  # 30
    ERROR = logging.ERROR  # 40


class OperationType(Enum):
    COMPUTE = "compute"
    MEMORY = "memory"
    COMPILATION = "compilation"
    SYNCHRONIZATION = "sync"
    CAPTURE = "capture"
    FUSION = "fusion"


@dataclass
class OperationEvent:
    timestamp: float
    operation_type: OperationType
    kernel: str
    time_ms: float
    phase: str
    memory_usage: Optional[float] = None
    thread_id: Optional[int] = None
    kernel_type: Optional[str] = None


class IrisLogger:
    _instance: Optional["IrisLogger"] = None

    def __init__(
        self, log_path: Optional[Path] = None, level: LogLevel = LogLevel.INFO
    ):
        if log_path is None:
            log_path = Path.home() / ".iris_cache" / "iris_log.jsonl"

        self.log_path = log_path
        self.log_path.parent.mkdir(exist_ok=True, parents=True)
        self.level = level
        self.session_start = time.time()
        self.operation_stack = []
        self.console_enabled = True
        self.file_enabled = True

        self.logger = logging.getLogger("iris")
        self.logger.setLevel(level.value)

        self.console_handler = logging.StreamHandler(sys.stdout)
        # Change the formatter to display the level name
        console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
        self.console_handler.setFormatter(console_formatter)
        self.logger.addHandler(self.console_handler)

        file_handler = logging.FileHandler(log_path.with_suffix(".log"))
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

    @classmethod
    def get_instance(cls, **kwargs) -> "IrisLogger":
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def configure(
        self,
        level: Optional[LogLevel] = None,
        console: Optional[bool] = None,
        file: Optional[bool] = None,
    ):
        if level is not None:
            self.level = level
            self.logger.setLevel(level.value)

        if console is not None:
            self.console_enabled = console
            if console and self.console_handler not in self.logger.handlers:
                self.logger.addHandler(self.console_handler)
            elif not console and self.console_handler in self.logger.handlers:
                self.logger.removeHandler(self.console_handler)

        if file is not None:
            self.file_enabled = file

    def debug(self, message: str, **kwargs):
        if self.level.value <= LogLevel.DEBUG.value:
            self._log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        if self.level.value <= LogLevel.INFO.value:
            self._log(LogLevel.INFO, message, **kwargs)

    def profile(self, message: str, **kwargs):
        if self.level.value <= LogLevel.PROFILE.value:
            self._log(LogLevel.PROFILE, message, **kwargs)

    def warning(self, message: str, **kwargs):
        if self.level.value <= LogLevel.WARNING.value:
            self._log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        if self.level.value <= LogLevel.ERROR.value:
            self._log(LogLevel.ERROR, message, **kwargs)

    def _log(self, level: LogLevel, message: str, **kwargs):
        if self.console_enabled:
            self.logger.log(level.value, message)
        if self.file_enabled:
            entry = {
                "timestamp": time.time(),
                "kernel": str(message),
                "phase": kwargs.get("phase", "run"),
                "time_ms": float(kwargs.get("time_ms", 0)),
            }
            if "memory_usage" in kwargs:
                entry["memory_usage"] = kwargs["memory_usage"]
            if "thread_id" in kwargs:
                entry["thread_id"] = kwargs["thread_id"]
            if "kernel_type" in kwargs:
                entry["kernel_type"] = kwargs["kernel_type"]
            self.log_path.parent.mkdir(exist_ok=True, parents=True)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")

    def start_operation(self, operation_type: OperationType, kernel: str,
                       phase: str = "run", metadata: Dict[str, Any] = None):
        event = OperationEvent(
            timestamp=time.time(),
            operation_type=operation_type,
            kernel=kernel,
            time_ms=0.0,
            phase=phase,
            memory_usage=metadata.get("memory_usage") if metadata else None,
            thread_id=metadata.get("thread_id") if metadata else None,
            kernel_type=metadata.get("kernel_type") if metadata else None
        )

        self.operation_stack.append(event)
        return len(self.operation_stack) - 1

    def end_operation(self, operation_id: int):
        if operation_id < len(self.operation_stack):
            event = self.operation_stack[operation_id]
            event.time_ms = (time.time() - event.timestamp) * 1000

            log_entry = {
                "timestamp": event.timestamp,
                "kernel": event.kernel,
                "time_ms": event.time_ms,
                "phase": event.phase
            }

            if event.memory_usage is not None:
                log_entry["memory_usage"] = event.memory_usage
            if event.thread_id is not None:
                log_entry["thread_id"] = event.thread_id
            if event.kernel_type is not None:
                log_entry["kernel_type"] = event.kernel_type

            # Use the main _log method to ensure console output
            self._log(LogLevel.PROFILE, f"{event.kernel} took {event.time_ms:.2f} ms")
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

    def log_memory_usage(self, kernel: str, size_mb: float):
        log_entry = {
            "timestamp": time.time(),
            "kernel": kernel,
            "time_ms": 0.0,
            "phase": "memory",
            "memory_usage": size_mb
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    @contextmanager
    def timed_operation(self, operation_type: OperationType, kernel: str,
                       phase: str = "run", metadata: Dict[str, Any] = None):
        op_id = self.start_operation(operation_type, kernel, phase, metadata)
        try:
            yield
        finally:
            self.end_operation(op_id)

def get_logger() -> IrisLogger:
    return IrisLogger.get_instance()


def configure_logging(
    level: LogLevel = LogLevel.INFO, console: bool = True, file: bool = True
):
    logger = get_logger()
    logger.configure(level=level, console=console, file=file)


# Unify the standalone log_event to use the logger class
def log_event(name: str, duration_ms: float, phase: str = "run"):
    """Convenience function to log an event using the global IrisLogger instance."""
    get_logger().profile(name, time_ms=duration_ms, phase=phase)
