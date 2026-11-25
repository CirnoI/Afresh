#!/usr/bin/env python3
"""Run the common one-shot iILS filtering executor with explicit inputs."""
from __future__ import annotations

# Bound BLAS before importing numpy/scipy. --threads belongs to the executor.
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'

from iilslib.one_shot_config import parse_args, normalize, inputs
from iilslib.one_shot import execute


def main(argv=None):
    args = parse_args(argv)
    specs = normalize(args)
    runs = inputs(args)
    execute(args, runs, specs)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
