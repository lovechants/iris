import pytest
import json
import time
import tempfile
from pathlib import Path
from metal_runtime.logging import IrisLogger, LogLevel, OperationType, get_logger, configure_logging

@pytest.fixture
def reset_logger():
    """Saves and restores the global logger's state."""
    logger = get_logger()
    original_level = logger.level
    original_console = logger.console_enabled
    original_file = logger.file_enabled
    yield 
    logger.configure(level=original_level, console=original_console, file=original_file)


def test_logger_creation(reset_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.jsonl"
        logger = IrisLogger(log_path=log_path)
        
        assert logger.log_path == log_path
        assert logger.level == LogLevel.INFO
        assert logger.console_enabled is True
        assert logger.file_enabled is True

def test_logger_configuration(reset_logger):
    logger = get_logger()
    
    configure_logging(level=LogLevel.DEBUG)
    assert logger.level == LogLevel.DEBUG
    
    configure_logging(console=False)
    assert logger.console_enabled is False
    
    configure_logging(file=False)
    assert logger.file_enabled is False

def test_operation_timing(reset_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.jsonl"
        logger = IrisLogger(log_path=log_path)
        
        op_id = logger.start_operation(OperationType.COMPUTE, "test_kernel", "test")
        time.sleep(0.01) 
        logger.end_operation(op_id)
        
        with open(log_path, 'r') as f:
            log_entry = json.loads(f.readline().strip())
        
        assert log_entry["kernel"] == "test_kernel"
        assert log_entry["phase"] == "test"
        assert log_entry["time_ms"] >= 10.0

def test_memory_logging(reset_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.jsonl"
        logger = IrisLogger(log_path=log_path)

        logger.log_memory_usage("test_kernel", 100.5)
        
        with open(log_path, 'r') as f:
            log_entry = json.loads(f.readline().strip())
        
        assert log_entry["kernel"] == "test_kernel"
        assert log_entry["phase"] == "memory"
        assert log_entry["memory_usage"] == 100.5

def test_timed_operation_context(reset_logger):
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.jsonl"
        logger = IrisLogger(log_path=log_path)
        
        with logger.timed_operation(OperationType.COMPUTE, "test_context", "ctx"):
            time.sleep(0.01) 
        
        with open(log_path, 'r') as f:
            log_entry = json.loads(f.readline().strip())
        
        assert log_entry["kernel"] == "test_context"
        assert log_entry["phase"] == "ctx"
        assert log_entry["time_ms"] >= 10.0
