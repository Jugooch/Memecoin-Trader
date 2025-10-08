"""
Structured logging setup for Frontrun Bot
Uses structlog for high-performance JSON logging
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from structlog.typing import EventDict, Processor


def add_timestamp(logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add ISO timestamp to log events"""
    from datetime import datetime, timezone
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_log_level(logger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict"""
    event_dict["level"] = method_name
    return event_dict


def setup_logging(
    level: str = "INFO",
    format: str = "json",
    output_file: Optional[str] = None
) -> None:
    """
    Configure structured logging for the application

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format: Output format ("json" or "console")
        output_file: Optional file path for log output
    """
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    # Ensure log directory exists if output_file specified
    if output_file:
        log_path = Path(output_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Add file handler
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(numeric_level)
        logging.root.addHandler(file_handler)

    # Configure structlog processors
    processors: list[Processor] = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        add_log_level,
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if format == "json":
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Console output for development
        # Try to use colors if colorama is available
        try:
            import colorama
            use_colors = True
        except ImportError:
            use_colors = False

        processors.extend([
            structlog.dev.ConsoleRenderer(
                colors=use_colors,
                exception_formatter=structlog.dev.plain_traceback,
            )
        ])

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a configured logger instance

    Args:
        name: Logger name (usually __name__)

    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


# Example usage
if __name__ == "__main__":
    # Setup logging
    setup_logging(level="DEBUG", format="console")

    # Get logger
    logger = get_logger(__name__)

    # Test logging
    logger.info("test_message", key1="value1", key2=42)
    logger.debug("debug_message", detail="some detail")
    logger.warning("warning_message", warning_type="test")
    logger.error("error_message", error_code=500)

    # Test with exception
    try:
        1 / 0
    except Exception as e:
        logger.error("exception_occurred", exc_info=True)
