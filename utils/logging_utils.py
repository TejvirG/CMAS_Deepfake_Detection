"""Consistent logging setup used across train/evaluate/experiment scripts."""
import logging
import os
import sys
from pathlib import Path


def get_logger(name: str, log_dir: str = "logs/", filename: str = "run.log") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:  # avoid duplicate handlers on re-import
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(os.path.join(log_dir, filename))
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
