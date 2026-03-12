import io
import sys
from pathlib import Path

import pytest
from loguru import logger

from mdu.utils.logging import configure_logger, get_logger


@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger configuration before and after each test."""
    # Remove all handlers before test
    logger.remove()
    yield
    # Reset to default after test
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="DEBUG",
        colorize=True,
    )


class TestGetLogger:
    """Test suite for get_logger function."""

    def test_get_logger_returns_logger_instance(self):
        """Test that get_logger returns a valid logger instance."""
        log = get_logger("test_module")
        assert log is not None
        # Check it has the expected loguru methods
        assert hasattr(log, "debug")
        assert hasattr(log, "info")
        assert hasattr(log, "warning")
        assert hasattr(log, "error")
        assert hasattr(log, "critical")

    def test_get_logger_with_different_names(self):
        """Test that get_logger can be called with different names."""
        log1 = get_logger("module1")
        log2 = get_logger("module2")
        assert log1 is not None
        assert log2 is not None
        # Both should be logger instances
        assert hasattr(log1, "info")
        assert hasattr(log2, "info")

    def test_get_logger_propagate_parameter(self):
        """Test that propagate parameter is accepted (for compatibility)."""
        log1 = get_logger("test", propagate=True)
        log2 = get_logger("test", propagate=False)
        assert log1 is not None
        assert log2 is not None

    def test_get_logger_log_level_parameter(self):
        """Test that log_level parameter is accepted."""
        log = get_logger("test", log_level="INFO")
        assert log is not None

    def test_logger_outputs_messages(self):
        """Test that logger actually outputs messages."""
        # Capture stderr
        output = io.StringIO()
        logger.remove()
        logger.add(output, format="{level} - {message}", colorize=False)

        log = get_logger("test_output")
        log.info("Test message")

        output_text = output.getvalue()
        assert "INFO" in output_text
        assert "Test message" in output_text

    def test_logger_multiple_levels(self):
        """Test that logger handles different log levels correctly."""
        output = io.StringIO()
        logger.remove()
        logger.add(output, format="{level} - {message}", colorize=False)

        log = get_logger("test_levels")
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")
        log.critical("Critical message")

        output_text = output.getvalue()
        assert "DEBUG" in output_text
        assert "INFO" in output_text
        assert "WARNING" in output_text
        assert "ERROR" in output_text
        assert "CRITICAL" in output_text
        assert "Debug message" in output_text
        assert "Info message" in output_text
        assert "Warning message" in output_text
        assert "Error message" in output_text
        assert "Critical message" in output_text


class TestConfigureLogger:
    """Test suite for configure_logger function."""

    def test_configure_logger_default_settings(self):
        """Test configure_logger with default settings."""
        output = io.StringIO()
        configure_logger(sink=output)

        log = get_logger("test")
        log.info("Test message")

        output_text = output.getvalue()
        assert "Test message" in output_text

    def test_configure_logger_level_filtering(self):
        """Test that configure_logger respects log level filtering."""
        output = io.StringIO()
        configure_logger(level="WARNING", sink=output, colorize=False)

        log = get_logger("test")
        log.debug("Debug message")
        log.info("Info message")
        log.warning("Warning message")
        log.error("Error message")

        output_text = output.getvalue()
        # DEBUG and INFO should be filtered out
        assert "Debug message" not in output_text
        assert "Info message" not in output_text
        # WARNING and ERROR should be present
        assert "Warning message" in output_text
        assert "Error message" in output_text

    def test_configure_logger_custom_format(self):
        """Test configure_logger with custom format string."""
        output = io.StringIO()
        custom_format = "{level} | {message}"
        configure_logger(format_string=custom_format, sink=output, colorize=False)

        log = get_logger("test")
        log.info("Custom format test")

        output_text = output.getvalue()
        assert "INFO | Custom format test" in output_text

    def test_configure_logger_without_colorize(self):
        """Test configure_logger with colorize=False."""
        output = io.StringIO()
        configure_logger(colorize=False, sink=output)

        log = get_logger("test")
        log.info("No color test")

        output_text = output.getvalue()
        # Should not contain ANSI color codes
        assert "\033[" not in output_text or "<" not in output_text
        assert "No color test" in output_text

    def test_configure_logger_file_sink(self, tmp_path):
        """Test configure_logger with file sink."""
        log_file = tmp_path / "test.log"
        configure_logger(sink=str(log_file), colorize=False)

        log = get_logger("test")
        log.info("File sink test")

        # Ensure the log is flushed
        logger.complete()

        # Read the log file
        assert log_file.exists()
        content = log_file.read_text()
        assert "File sink test" in content

    def test_configure_logger_multiple_calls(self):
        """Test that calling configure_logger multiple times resets configuration."""
        output1 = io.StringIO()
        configure_logger(level="INFO", sink=output1, colorize=False)

        log = get_logger("test")
        log.debug("Should not appear")
        log.info("Should appear in output1")

        output2 = io.StringIO()
        configure_logger(level="DEBUG", sink=output2, colorize=False)

        log.debug("Should appear in output2")
        log.info("Also in output2")

        # First output should only have INFO
        output1_text = output1.getvalue()
        assert "Should not appear" not in output1_text
        assert "Should appear in output1" in output1_text

        # Second output should have both DEBUG and INFO
        output2_text = output2.getvalue()
        assert "Should appear in output2" in output2_text
        assert "Also in output2" in output2_text


class TestLoggerIntegration:
    """Integration tests for logger functionality."""

    def test_multiple_loggers_same_output(self):
        """Test that multiple loggers write to the same output."""
        output = io.StringIO()
        configure_logger(sink=output, colorize=False)

        log1 = get_logger("module1")
        log2 = get_logger("module2")

        log1.info("Message from module1")
        log2.info("Message from module2")

        output_text = output.getvalue()
        assert "Message from module1" in output_text
        assert "Message from module2" in output_text

    def test_logger_context_binding(self):
        """Test that logger name is bound to context."""
        output = io.StringIO()
        configure_logger(
            format_string="{extra[name]} - {message}",
            sink=output,
            colorize=False,
        )

        log = get_logger("my_module")
        log.info("Context test")

        output_text = output.getvalue()
        assert "my_module" in output_text
        assert "Context test" in output_text

    def test_exception_logging(self):
        """Test that exceptions are logged properly."""
        output = io.StringIO()
        configure_logger(sink=output, colorize=False)

        log = get_logger("test")

        try:
            raise ValueError("Test exception")
        except ValueError:
            log.exception("An error occurred")

        output_text = output.getvalue()
        assert "An error occurred" in output_text
        assert "ValueError" in output_text
        assert "Test exception" in output_text

    def test_logger_with_formatting_arguments(self):
        """Test logger with formatting arguments."""
        output = io.StringIO()
        configure_logger(sink=output, colorize=False, format_string="{message}")

        log = get_logger("test")
        name = "Alice"
        age = 30
        log.info(f"User {name} is {age} years old")

        output_text = output.getvalue()
        assert "User Alice is 30 years old" in output_text
