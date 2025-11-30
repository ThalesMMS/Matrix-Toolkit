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


# ---------------------------------------------------------------------------
# Additional Matrix Operations
# ---------------------------------------------------------------------------


def identity_matrix(size: int) -> Matrix:
    """Create an identity matrix of given size."""
    if size <= 0:
        raise ValueError("O tamanho deve ser positivo.")
    return [
        [Fraction(1) if i == j else Fraction(0) for j in range(size)]
        for i in range(size)
    ]


def zero_matrix(rows: int, cols: int) -> Matrix:
    """Create a zero matrix with given dimensions."""
    if rows <= 0 or cols <= 0:
        raise ValueError("As dimensoes devem ser positivas.")
    return [[Fraction(0) for _ in range(cols)] for _ in range(rows)]


def diagonal(matrix: Sequence[Sequence]) -> List[Fraction]:
    """Extract the main diagonal of the matrix."""
    rows, cols = _validate_rectangular(matrix)
    matrix_f = _clone_matrix(matrix)
    return [matrix_f[i][i] for i in range(min(rows, cols))]


def trace(matrix: Sequence[Sequence]) -> Fraction:
    """Compute the trace (sum of diagonal elements) of a square matrix."""
    _validate_square(matrix)
    return sum(diagonal(matrix), Fraction(0))


def is_symmetric(matrix: Sequence[Sequence]) -> bool:
    """Check if the matrix is symmetric (equal to its transpose)."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)
    for i in range(size):
        for j in range(i + 1, size):
            if matrix_f[i][j] != matrix_f[j][i]:
                return False
    return True


def is_diagonal(matrix: Sequence[Sequence]) -> bool:
    """Check if the matrix is diagonal (all off-diagonal entries are zero)."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)
    for i in range(size):
        for j in range(size):
            if i != j and matrix_f[i][j] != 0:
                return False
    return True


def is_identity(matrix: Sequence[Sequence]) -> bool:
    """Check if the matrix is an identity matrix."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)
    for i in range(size):
        for j in range(size):
            expected = Fraction(1) if i == j else Fraction(0)
            if matrix_f[i][j] != expected:
                return False
    return True


def is_zero(matrix: Sequence[Sequence]) -> bool:
    """Check if all entries in the matrix are zero."""
    _validate_rectangular(matrix)
    matrix_f = _clone_matrix(matrix)
    for row in matrix_f:
        for value in row:
            if value != 0:
                return False
    return True


def is_upper_triangular(matrix: Sequence[Sequence]) -> bool:
    """Check if the matrix is upper triangular (zeros below main diagonal)."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)
    for i in range(size):
        for j in range(i):
            if matrix_f[i][j] != 0:
                return False
    return True


def is_lower_triangular(matrix: Sequence[Sequence]) -> bool:
    """Check if the matrix is lower triangular (zeros above main diagonal)."""
    size = _validate_square(matrix)
    matrix_f = _clone_matrix(matrix)
    for i in range(size):
        for j in range(i + 1, size):
            if matrix_f[i][j] != 0:
                return False
    return True


def matrix_power(matrix: Sequence[Sequence], exponent: int) -> Matrix:
    """Raise a square matrix to an integer power (non-negative or -1 for inverse)."""
    size = _validate_square(matrix)

    if exponent < -1:
        raise ValueError("Expoentes menores que -1 nao sao suportados diretamente.")

    if exponent == -1:
        return inverse(matrix)

    if exponent == 0:
        return identity_matrix(size)

    # Binary exponentiation for efficiency.
    result = identity_matrix(size)
    base = _clone_matrix(matrix)

    power = exponent
    while power > 0:
        if power % 2 == 1:
            result = multiply_matrices(result, base)
        base = multiply_matrices(base, base)
        power //= 2

    return result


def rank(matrix: Sequence[Sequence]) -> int:
    """Compute the rank of a matrix (number of non-zero rows in RREF)."""
    reduced = rref(matrix)
    count = 0
    for row in reduced:
        if any(value != 0 for value in row):
            count += 1
    return count


def nullity(matrix: Sequence[Sequence]) -> int:
    """Compute the nullity of a matrix (number of columns minus rank)."""
    rows, cols = _validate_rectangular(matrix)
    return cols - rank(matrix)


def lu_decomposition(matrix: Sequence[Sequence]) -> Tuple[Matrix, Matrix]:
    """
    Compute the LU decomposition of a square matrix (without pivoting).

    Returns (L, U) where A = L * U, L is lower triangular with ones on the
    diagonal, and U is upper triangular.

    Raises an error if a zero pivot is encountered (pivoting not supported).
    """
    size = _validate_square(matrix)
    L = identity_matrix(size)
    U = _clone_matrix(matrix)

    for col in range(size):
        if U[col][col] == 0:
            raise ValueError(
                "Decomposicao LU sem pivoteamento falhou; encontrou pivo zero."
            )
        for row in range(col + 1, size):
            factor = U[row][col] / U[col][col]
            L[row][col] = factor
            for k in range(col, size):
                U[row][k] -= factor * U[col][k]

    return L, U


def lu_decomposition_with_steps(
    matrix: Sequence[Sequence],
) -> Tuple[Matrix, Matrix, StepLog]:
    """LU decomposition with step-by-step log for educational display."""
    size = _validate_square(matrix)
    L = identity_matrix(size)
    U = _clone_matrix(matrix)
    steps: StepLog = [("Matriz inicial U", _clone_matrix(U))]

    for col in range(size):
        if U[col][col] == 0:
            raise ValueError(
                "Decomposicao LU sem pivoteamento falhou; encontrou pivo zero."
            )
        for row in range(col + 1, size):
            factor = U[row][col] / U[col][col]
            L[row][col] = factor
            for k in range(col, size):
                U[row][k] -= factor * U[col][k]
            steps.append(
                (
                    f"L[{row+1}][{col+1}] = {_format_fraction(factor)}, "
                    f"U: R{row+1} = R{row+1} - {_format_fraction(factor)} * R{col+1}",
                    _clone_matrix(U),
                )
            )

    steps.append(("Matriz L final", _clone_matrix(L)))
    steps.append(("Matriz U final", _clone_matrix(U)))
    return L, U, steps


def solve_system(A: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """
    Solve the linear system Ax = b using RREF on the augmented matrix.

    Returns the solution vector as a column matrix.
    Raises an error if the system has no unique solution.
    """
    rows_a, cols_a = _validate_rectangular(A)
    rows_b, cols_b = _validate_rectangular(b)

    if rows_a != rows_b:
        raise ValueError("O numero de linhas de A deve coincidir com o de b.")
    if cols_b != 1:
        raise ValueError("b deve ser uma matriz coluna (uma coluna).")

    # Build augmented matrix [A | b].
    augmented = [
        [_to_fraction(val) for val in row_a] + [_to_fraction(b[i][0])]
        for i, row_a in enumerate(A)
    ]

    reduced = rref(augmented)

    # Check for inconsistency or free variables.
    for i, row in enumerate(reduced):
        # All coefficients zero but constant nonzero means no solution.
        if all(val == 0 for val in row[:-1]) and row[-1] != 0:
            raise ValueError("O sistema nao possui solucao (inconsistente).")

    # For unique solution, we need exactly cols_a pivots.
    pivot_count = rank(A)
    if pivot_count < cols_a:
        raise ValueError(
            "O sistema possui infinitas solucoes (variaveis livres presentes)."
        )

    # Extract solution from the last column.
    return [[reduced[i][-1]] for i in range(cols_a)]


def solve_system_with_steps(
    A: Sequence[Sequence], b: Sequence[Sequence]
) -> Tuple[Matrix, StepLog]:
    """Solve a linear system with step-by-step log."""
    rows_a, cols_a = _validate_rectangular(A)
    rows_b, cols_b = _validate_rectangular(b)

    if rows_a != rows_b:
        raise ValueError("O numero de linhas de A deve coincidir com o de b.")
    if cols_b != 1:
        raise ValueError("b deve ser uma matriz coluna (uma coluna).")

    augmented = [
        [_to_fraction(val) for val in row_a] + [_to_fraction(b[i][0])]
        for i, row_a in enumerate(A)
    ]

    reduced, steps = rref_with_steps(augmented)

    for i, row in enumerate(reduced):
        if all(val == 0 for val in row[:-1]) and row[-1] != 0:
            raise ValueError("O sistema nao possui solucao (inconsistente).")

    pivot_count = rank(A)
    if pivot_count < cols_a:
        raise ValueError(
            "O sistema possui infinitas solucoes (variaveis livres presentes)."
        )

    solution = [[reduced[i][-1]] for i in range(cols_a)]
    steps.append(("Solucao x", solution))
    return solution, steps


def hadamard_product(a: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """
    Element-wise (Hadamard) product of two matrices.

    Both matrices must have the same dimensions.
    """
    rows, cols = _validate_same_shape(a, b)
    a_f = _clone_matrix(a)
    b_f = _clone_matrix(b)
    return [[a_f[i][j] * b_f[i][j] for j in range(cols)] for i in range(rows)]


def frobenius_norm_squared(matrix: Sequence[Sequence]) -> Fraction:
    """
    Compute the squared Frobenius norm (sum of squares of all entries).

    Returns a Fraction; take the square root externally for the actual norm.
    """
    _validate_rectangular(matrix)
    matrix_f = _clone_matrix(matrix)
    total = Fraction(0)
    for row in matrix_f:
        for value in row:
            total += value * value
    return total


def minor(matrix: Sequence[Sequence], row: int, col: int) -> Fraction:
    """
    Compute the (row, col) minor of a square matrix.

    The minor M_ij is the determinant of the submatrix formed by deleting
    row i and column j.
    """
    size = _validate_square(matrix)
    if row < 0 or row >= size or col < 0 or col >= size:
        raise ValueError("Indices de linha e coluna devem estar dentro da matriz.")
    matrix_f = _clone_matrix(matrix)
    submatrix = [
        [matrix_f[r][c] for c in range(size) if c != col]
        for r in range(size)
        if r != row
    ]
    return determinant(submatrix)


def cofactor(matrix: Sequence[Sequence], row: int, col: int) -> Fraction:
    """
    Compute the (row, col) cofactor of a square matrix.

    Cofactor C_ij = (-1)^(i+j) * M_ij where M_ij is the minor.
    """
    sign = Fraction((-1) ** (row + col))
    return sign * minor(matrix, row, col)


def submatrix(
    matrix: Sequence[Sequence],
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
) -> Matrix:
    """
    Extract a submatrix from the given matrix.

    Indices are 0-based; end indices are exclusive.
    """
    _validate_rectangular(matrix)
    matrix_f = _clone_matrix(matrix)
    return [row[start_col:end_col] for row in matrix_f[start_row:end_row]]


def concatenate_horizontal(a: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """Concatenate two matrices horizontally (side by side)."""
    rows_a, _ = _validate_rectangular(a)
    rows_b, _ = _validate_rectangular(b)
    if rows_a != rows_b:
        raise ValueError("As matrizes devem ter o mesmo numero de linhas.")
    a_f = _clone_matrix(a)
    b_f = _clone_matrix(b)
    return [row_a + row_b for row_a, row_b in zip(a_f, b_f)]


def concatenate_vertical(a: Sequence[Sequence], b: Sequence[Sequence]) -> Matrix:
    """Concatenate two matrices vertically (stacked)."""
    _, cols_a = _validate_rectangular(a)
    _, cols_b = _validate_rectangular(b)
    if cols_a != cols_b:
        raise ValueError("As matrizes devem ter o mesmo numero de colunas.")
    return _clone_matrix(a) + _clone_matrix(b)
