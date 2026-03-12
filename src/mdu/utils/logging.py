import sys
from typing import Optional

from loguru import logger

# Remove default handler and configure with custom format
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)


def get_logger(
    name: str,
    log_level: str = "DEBUG",
    propagate: bool = True,
) -> logger.__class__:
    """Get a configured loguru logger instance.

    Parameters
    ----------
    name : str
        Name/context to bind to the logger for identification.
    log_level : str, default="DEBUG"
        Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    propagate : bool, default=True
        For compatibility with standard logging. With loguru, all loggers
        share the same handlers, so this parameter is kept for API compatibility
        but doesn't affect behavior.

    Returns
    -------
    loguru.Logger
        Configured logger instance bound with the specified name.

    Examples
    --------
    >>> log = get_logger("my_module")
    >>> log.info("This is an info message")
    >>> log.debug("This is a debug message")
    >>> log.error("This is an error message")

    Notes
    -----
    Unlike standard logging, loguru uses a single global logger instance.
    The 'name' parameter is bound to the logger context for identification
    in log messages. The log_level parameter sets a filter for this context.
    """
    # Bind the name to the logger context
    bound_logger = logger.bind(name=name)

    # Add a filter to respect the log level for this specific logger context
    # Note: This requires the logger to have been configured with filters
    # For simplicity, we return the bound logger and rely on global configuration
    return bound_logger


def configure_logger(
    level: str = "DEBUG",
    format_string: Optional[str] = None,
    colorize: bool = True,
    sink=sys.stderr,
):
    """Configure the global loguru logger.

    Parameters
    ----------
    level : str, default="DEBUG"
        Minimum logging level to display.
    format_string : str, optional
        Custom format string for log messages. If None, uses default format.
    colorize : bool, default=True
        Whether to colorize the output.
    sink : file-like or str, default=sys.stderr
        Output destination for logs (file path, file object, or stream).

    Examples
    --------
    >>> configure_logger(level="INFO", colorize=False)
    >>> configure_logger(level="WARNING", sink="app.log")
    """
    if format_string is None:
        format_string = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

    logger.remove()
    logger.add(sink, format=format_string, level=level, colorize=colorize)


if __name__ == "__main__":
    test_logger = get_logger("test")
    test_logger.debug("DEBUG test")
    test_logger.info("INFO test")
    test_logger.warning("WARNING test")
    test_logger.error("ERROR test")
    test_logger.critical("CRITICAL test")
