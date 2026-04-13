"""
Pharma-Z Logging Configuration

Structured logging setup for the entire application.
"""

import logging
import sys


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure and return the application logger."""

    logger = logging.getLogger("pharma_z")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# Pre-configured logger — import and use directly
logger = setup_logging()
