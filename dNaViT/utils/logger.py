# Copyright (c) 2026 LongCat dNaViT Team.
# SPDX-License-Identifier: Apache-2.0

"""This files contains util functions supporting logging to terminal and files."""

import functools
import sys
from accelerate.logging import MultiProcessAdapter
import logging
from termcolor import colored

from iopath.common.file_io import PathManager as PathManagerClass

__all__ = ["setup_logger", "PathManager"]

PathManager = PathManagerClass()


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", self._root_name)
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


@functools.lru_cache()
def setup_logger(name="dNaViT", log_level: str = None, color=True, use_accelerate=True,
                 output_file=None):
    logger = logging.getLogger(name)
    if log_level is None:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(log_level.upper())

    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_file is not None:
        fileHandler = logging.FileHandler(output_file)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    if use_accelerate:
        return MultiProcessAdapter(logger, {})
    else:
        return logger