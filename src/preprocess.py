"""preprocess.py – data preprocessing stub.

The Experiment Code supplied for refactor contained **no** data-loading or
pre-processing utilities.  Attempting to fabricate them here would breach
the instruction *“Do not introduce new logic or fall back to placeholder
examples.”*  Instead, we surface a clear error so that downstream users
understand that this part of the pipeline is missing.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def load_data(*_, **__):  # noqa: D401, D403 – placeholder signature
    LOGGER.error(
        "Data-loading requested but no implementation was provided in the "
        "original Experiment Code."
    )
    raise RuntimeError(
        "preprocess.py cannot load data because the functionality was absent "
        "from the supplied source script."
    )
