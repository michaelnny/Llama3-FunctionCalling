# Copyright (c) 2023 Michael Hu.
# This project is released under the MIT License.
# See the accompanying LICENSE file for details.

from typing import Optional
import sys
import logging

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class DummyLogger:
    def __init__(self):
        pass

    def _noop(self, *args, **kwargs):
        pass

    info = warning = debug = _noop


def create_logger(level='INFO', rank=0) -> logging.Logger:
    if rank == 0:
        handler = logging.StreamHandler(stream=sys.stderr)
        formatter = logging.Formatter(
            fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        level = str(level).upper()
        if level == 'DEBUG':
            veb = logging.DEBUG
        elif level == 'ERROR':
            veb = logging.ERROR
        else:
            veb = logging.INFO
        logger.setLevel(veb)
        logger.addHandler(handler)

        return logger
    else:
        return DummyLogger()


def log_metrics(writer: SummaryWriter, step: int, metrics: dict):
    if writer is None:
        return
    elif not isinstance(metrics, dict):
        return

    for key, value in metrics.items():
        try:
            writer.add_scalar(key, value, step)
        except Exception as error:
            logger.error(f'Failed to log metrics, error {str(error)}')


def pretty_print(metrics: dict, key_width: int = 40, value_width: int = 10):
    """Print a two column-table consists metrics key and values to console"""

    metrics = prettify_metrics(metrics)

    # Print header
    print(f"{'Metric'.ljust(key_width)} {'Value'.ljust(value_width)}")
    print('-' * (key_width + value_width))

    # Print each metric
    for key, value in metrics.items():
        print(f'{key.ljust(key_width)} {str(value).ljust(value_width)}')


def format_value(v):
    if isinstance(v, float):
        # Check if the value is very small or very large and use scientific notation if needed
        if abs(v) < 1e-4 or abs(v) > 1e4:
            return f'{v:.4e}'  # Scientific notation with 4 decimal places
        else:
            return f'{v:.4f}'  # Fixed-point notation with 4 decimal places
    return v


def prettify_metrics(metrics: dict) -> dict:
    formatted_metrics = {}
    for k, v in metrics.items():
        formatted_metrics[k] = format_value(v)
    return formatted_metrics
