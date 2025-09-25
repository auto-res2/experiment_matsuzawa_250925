"""train.py – extracted from the original (non-existent) experiment.
Since the source contained no runnable training logic, this file only
exposes a stub that notifies the caller that the training procedure was
missing from the provided script.  Importing code that never existed
would raise runtime errors downstream; therefore we guard the call with a
RuntimeError that explains the situation clearly so that users of the
package are immediately aware that there is nothing to execute.

Keeping such an explicit guard is preferable to silently returning `pass`
as it fails fast and prevents time-consuming debugging sessions.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger(__name__)


def train(*args, **kwargs):  # noqa: D401, D403 – minimal signature on purpose
    """Dummy entry-point because the original script contained no code.

    Raises
    ------
    RuntimeError
        Always raised to inform the caller that no training logic was
        supplied in the *Experiment Code* section that this repository
        has been generated from.
    """

    LOGGER.error(
        "The original Experiment Code did not contain any training logic."
    )
    raise RuntimeError(
        "train.py cannot run because no training algorithm was available in "
        "the provided source script."
    )
