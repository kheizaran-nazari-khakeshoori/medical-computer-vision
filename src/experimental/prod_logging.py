"""Configuring logging for production."""

import logging

from src.logger import get_logger


def configure_prod_logging():
    logger = get_logger("prod", log_file="logs/prod.log")
    logger.setLevel(logging.WARNING)
    return logger
