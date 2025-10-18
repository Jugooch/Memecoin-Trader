"""
Logging configuration for the trading bot
"""

import logging
import os
from datetime import datetime


def setup_logging(level: str = "INFO", log_file: str = "logs/trading.log"):
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # File handler - use UTF-8 encoding for emoji support
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)

    # Console handler - use UTF-8 with error handling for Windows
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)

    # On Windows, handle emoji characters gracefully
    import sys
    import io
    if sys.platform == 'win32':
        # Wrap the console stream to handle Unicode errors
        if hasattr(console_handler.stream, 'buffer'):
            # Use UTF-8 encoding with backslashreplace for unsupported characters
            console_handler.stream = io.TextIOWrapper(
                console_handler.stream.buffer,
                encoding='utf-8',
                errors='backslashreplace',
                line_buffering=True
            )

    root_logger.addHandler(console_handler)
    
    # Create main logger
    logger = logging.getLogger("MemecoinBot")
    logger.info(f"Logging initialized - Level: {level}, File: {log_file}")
    
    # Suppress noisy GQL transport logs
    logging.getLogger("gql.transport.aiohttp").setLevel(logging.WARNING)
    logging.getLogger("gql.transport.websockets").setLevel(logging.WARNING)
    logging.getLogger("gql.transport").setLevel(logging.WARNING)
    
    return logger