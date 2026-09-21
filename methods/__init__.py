"""Baseline method implementations for GPAT-TransferBench M6.

Each subpackage implements exactly one frozen M6B method contract. Nothing here re-declares a
scientific parameter: every value is read from `configs/methods/<method>.yaml`, which is verified
byte-identical against `frozen_config_snapshot/` before use.
"""
