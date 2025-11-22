#
# __init__.py
# Matrix Toolkit
#
# Exposes the high-level matrix operations so consumers can import directly
# from matrix_toolkit without touching submodules.
#
# Thales Matheus Mendonça Santos - November 2025

"""Matrix Toolkit core package."""

# Re-export the high-level operations so users can import them directly from
# `matrix_toolkit` instead of digging into submodules.

from .operations import (  # noqa: F401
    add_matrices,
    adjugate,
    cofactor_matrix,
    determinant,
    inverse,
    multiply_matrices,
    rref,
    rref_with_steps,
    scalar_multiply,
    subtract_matrices,
    transpose,
)
