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
    # Basic operations
    add_matrices,
    subtract_matrices,
    scalar_multiply,
    multiply_matrices,
    hadamard_product,
    transpose,
    # Determinant and inverse
    determinant,
    cofactor_matrix,
    adjugate,
    inverse,
    minor,
    cofactor,
    # Matrix reduction
    rref,
    rref_with_steps,
    # Matrix generators
    identity_matrix,
    zero_matrix,
    # Diagonal and trace
    diagonal,
    trace,
    # Matrix properties
    is_symmetric,
    is_diagonal,
    is_identity,
    is_zero,
    is_upper_triangular,
    is_lower_triangular,
    # Matrix power
    matrix_power,
    # Rank and nullity
    rank,
    nullity,
    # LU decomposition
    lu_decomposition,
    lu_decomposition_with_steps,
    # Linear systems
    solve_system,
    solve_system_with_steps,
    # Norms
    frobenius_norm_squared,
    # Submatrix operations
    submatrix,
    concatenate_horizontal,
    concatenate_vertical,
)
