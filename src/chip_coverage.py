"""Compatibility shim: multitask-dit-quant's load_op_emulator("chip_coverage") imports this name.
The module lives in chip/chip_coverage.py."""
from chip.chip_coverage import *  # noqa: F401,F403
from chip.chip_coverage import MtditEmulator, make_mtdit_op_emulator  # noqa: F401
