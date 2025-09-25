"""evaluate.py – no evaluation logic available.

As with *train.py*, the source material contained no evaluation section.
This stub fails loudly so that continuous-integration pipelines will
immediately surface the missing implementation instead of producing false
success signals.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def evaluate(*args, **kwargs):  # noqa: D401, D403 – placeholder signature
    LOGGER.error(
        "The original Experiment Code did not contain any evaluation logic."
    )
    raise RuntimeError(
        "evaluate.py cannot run because no evaluation algorithm was present "
        "in the provided Experiment Code."
    )
