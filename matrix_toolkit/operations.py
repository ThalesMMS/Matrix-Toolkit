#
# operations.py
# Matrix Toolkit
#
# Core linear algebra routines (determinant, inverses, RREF, etc.) implemented
# with Fraction for exact arithmetic so results stay precise during calculations.
#
# Thales Matheus Mendonça Santos - November 2025

"""Pure matrix operations built on fractions.Fraction."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
from typing import List, Sequence, Tuple

Matrix = List[List[Fraction]]
StepLog = List[Tuple[str, Matrix]]


def _to_fraction(value) -> Fraction:
    """Convert numbers or stringy numbers to Fraction."""
    return value if isinstance(value, Fraction) else Fraction(value)


def _clone_matrix(matrix: Sequence[Sequence]) -> Matrix:
    """Return a deep copy of the matrix, coercing entries to Fraction."""
    # Coercing up front keeps downstream logic simple and avoids repeated conversions.
    return [[_to_fraction(value) for value in row] for row in matrix]


def _validate_rectangular(matrix: Sequence[Sequence]) -> Tuple[int, int]:
    """Ensure the matrix is non-empty and rectangular, returning (rows, cols)."""
    if not matrix:
        raise ValueError("A matriz nao pode ser vazia.")
    row_length = len(matrix[0])
    for row in matrix:
        if len(row) != row_length:
            raise ValueError("Todas as linhas da matriz precisam ter o mesmo tamanho.")
    return len(matrix), row_length


def _validate_same_shape(a: Sequence[Sequence], b: Sequence[Sequence]) -> Tuple[int, int]:
    """Ensure matrices share the same dimensions."""
    rows_a, cols_a = _validate_rectangular(a)
    rows_b, cols_b = _validate_rectangular(b)
    if rows_a != rows_b or cols_a != cols_b:
        raise ValueError("As matrizes precisam ter as mesmas dimensoes.")
    return rows_a, cols_a


def _validate_square(matrix: Sequence[Sequence]) -> int:
    """Ensure the matrix is square and return its order."""
    rows, cols = _validate_rectangular(matrix)
    if rows != cols:
        raise ValueError("A matriz deve ser quadrada.")
    return rows


def add_matrices(a: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """Return the sum A + B."""
    rows, cols = _validate_same_shape(a, b)
    return [
        [_to_fraction(x) + _to_fraction(y) for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)  # Pairwise addition on each row.
    ]


def subtract_matrices(a: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """Return the difference A - B."""
    rows, cols = _validate_same_shape(a, b)
    return [
        [_to_fraction(x) - _to_fraction(y) for x, y in zip(row_a, row_b)]
        for row_a, row_b in zip(a, b)  # Pairwise subtraction on each row.
    ]


def scalar_multiply(matrix: Sequence[Sequence], scalar) -> Matrix:
    """Multiply every entry by the provided scalar."""
    _validate_rectangular(matrix)
    scalar_fraction = _to_fraction(scalar)
    return [[_to_fraction(value) * scalar_fraction for value in row] for row in matrix]


def multiply_matrices(a: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """Return the matrix product A * B."""
    rows_a, cols_a = _validate_rectangular(a)
    rows_b, cols_b = _validate_rectangular(b)
    if cols_a != rows_b:
        raise ValueError("O numero de colunas de A deve ser igual ao numero de linhas de B.")

    a_f = _clone_matrix(a)
    b_f = _clone_matrix(b)
    result: Matrix = []
    for i in range(rows_a):
        row_result: List[Fraction] = []
        for j in range(cols_b):
            total = Fraction(0, 1)  # Start accumulation for position (i, j).
            for k in range(cols_a):
                # Standard dot product: row i of A with column j of B.
                total += a_f[i][k] * b_f[k][j]
            row_result.append(total)
        result.append(row_result)
    return result


def transpose(matrix: Sequence[Sequence]) -> Matrix:
    """Return the transpose of a matrix."""
    _validate_rectangular(matrix)
    return [list(column) for column in zip(*matrix)]


def determinant(matrix: Sequence[Sequence]) -> Fraction:
    """Compute the determinant recursively via Laplace expansion."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)

    if size == 1:
        return matrix_f[0][0]
    if size == 2:
        return matrix_f[0][0] * matrix_f[1][1] - matrix_f[1][0] * matrix_f[0][1]

    det = Fraction(0, 1)
    for column in range(size):
        sign = Fraction((-1) ** column, 1)
        submatrix = [
            [matrix_f[r][c] for c in range(size) if c != column] for r in range(1, size)
        ]  # Remove first row and current column to build minor matrix.
        # Expand along the first row: element * cofactor (sign * det(minor)).
        det += matrix_f[0][column] * sign * determinant(submatrix)
    return det


def cofactor_matrix(matrix: Sequence[Sequence]) -> Matrix:
    """Return the matrix of cofactors."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)
    cofactors: Matrix = []
    for row in range(size):
        cofactor_row: List[Fraction] = []
        for column in range(size):
            submatrix = [
                [matrix_f[r][c] for c in range(size) if c != column]
                for r in range(size)
                if r != row
            ]
            sign = Fraction((-1) ** (row + column), 1)
            cofactor_row.append(sign * determinant(submatrix))
        cofactors.append(cofactor_row)
    return cofactors


def adjugate(matrix: Sequence[Sequence]) -> Matrix:
    """Return the adjugate (transpose of the cofactor matrix)."""
    # The adjugate is the transposed cofactor matrix and is key to computing inverses.
    return transpose(cofactor_matrix(matrix))


def inverse(matrix: Sequence[Sequence]) -> Matrix:
    """Return the inverse matrix, raising an error if singular."""
    det = determinant(matrix)
    if det == 0:
        raise ValueError("A matriz eh singular; o determinante eh zero.")

    adj = adjugate(matrix)
    det_inverse = Fraction(1, 1) / det
    # Multiply each entry of the adjugate by 1/det to finish A^{-1}.
    return [[entry * det_inverse for entry in row] for row in adj]


def _format_fraction(value: Fraction) -> str:
    """Return a human friendly string for a Fraction."""
    return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(
        value.numerator
    )


def rref_with_steps(matrix: Sequence[Sequence]) -> Tuple[Matrix, StepLog]:
    """
    Reduce the matrix to RREF with Gauss–Jordan elimination.

    Returns both the final matrix and the logged steps for presentation.
    """
    rows, cols = _validate_rectangular(matrix)
    work = _clone_matrix(matrix)
    steps: StepLog = [("Matriz inicial", _clone_matrix(work))]  # Keep the starting point for display.
    pivot_row = 0  # Tracks which row should receive the next pivot.

    for col in range(cols):
        # Find pivot row.
        selected = None
        for row in range(pivot_row, rows):
            if work[row][col] != 0:
                selected = row
                break
        if selected is None:
            # No pivot in this column; move to next column.
            continue

        # Swap into place.
        if selected != pivot_row:
            work[pivot_row], work[selected] = work[selected], work[pivot_row]
            steps.append(
                (f"Troca R{pivot_row + 1} <-> R{selected + 1}", _clone_matrix(work))
            )

        pivot_value = work[pivot_row][col]
        if pivot_value != 1:
            factor = Fraction(1, 1) / pivot_value
            # Normalize the pivot row so the pivot entry becomes 1.
            work[pivot_row] = [value * factor for value in work[pivot_row]]
            steps.append(
                (
                    f"R{pivot_row + 1} = {_format_fraction(factor)} * R{pivot_row + 1}",
                    _clone_matrix(work),
                )
            )

        # Eliminate other rows.
        for row in range(rows):
            if row == pivot_row or work[row][col] == 0:
                continue
            factor = work[row][col]
            work[row] = [a - factor * b for a, b in zip(work[row], work[pivot_row])]  # Row_i <- Row_i - factor * pivot row.
            steps.append(
                (
                    f"R{row + 1} = R{row + 1} - {_format_fraction(factor)} * R{pivot_row + 1}",
                    _clone_matrix(work),
                )
            )

        pivot_row += 1  # Move pivot down one row before scanning next column.
        if pivot_row == rows:
            break

    steps.append(("Forma escalonada reduzida (RREF)", _clone_matrix(work)))
    return work, steps


def rref(matrix: Sequence[Sequence]) -> Matrix:
    """Return only the final RREF matrix."""
    result, _ = rref_with_steps(matrix)
    return result
