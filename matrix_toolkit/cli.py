#
# cli.py
# Matrix Toolkit
#
# Text-based menu for performing common matrix operations interactively,
# delegating computation to the core library and showing intermediate steps.
#
# Thales Matheus Mendonça Santos - November 2025

"""Interactive CLI for common matrix operations."""

from __future__ import annotations

import sys
from typing import Callable, Dict, List, Tuple

from . import (
    add_matrices,
    adjugate,
    cofactor_matrix,
    determinant,
    diagonal,
    frobenius_norm_squared,
    hadamard_product,
    identity_matrix,
    inverse,
    is_diagonal,
    is_identity,
    is_lower_triangular,
    is_symmetric,
    is_upper_triangular,
    is_zero,
    lu_decomposition_with_steps,
    matrix_power,
    multiply_matrices,
    nullity,
    rank,
    rref_with_steps,
    scalar_multiply,
    solve_system_with_steps,
    subtract_matrices,
    trace,
    transpose,
    zero_matrix,
)
from . import interactive as ui


# ---------------------------------------------------------------------------
# ANSI Color Support
# ---------------------------------------------------------------------------

class Colors:
    """ANSI escape codes for terminal colors (disabled if not a TTY)."""

    ENABLED = sys.stdout.isatty()

    RESET = "\033[0m" if ENABLED else ""
    BOLD = "\033[1m" if ENABLED else ""
    DIM = "\033[2m" if ENABLED else ""

    # Foreground colors
    RED = "\033[31m" if ENABLED else ""
    GREEN = "\033[32m" if ENABLED else ""
    YELLOW = "\033[33m" if ENABLED else ""
    BLUE = "\033[34m" if ENABLED else ""
    MAGENTA = "\033[35m" if ENABLED else ""
    CYAN = "\033[36m" if ENABLED else ""
    WHITE = "\033[37m" if ENABLED else ""


def _color(text: str, *codes: str) -> str:
    """Apply color codes to text."""
    return f"{''.join(codes)}{text}{Colors.RESET}"


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    """Print a styled section header."""
    border = "═" * (len(title) + 4)
    print()
    print(_color(border, Colors.CYAN))
    print(_color(f"  {title}  ", Colors.CYAN, Colors.BOLD))
    print(_color(border, Colors.CYAN))
    print()


def _subheader(text: str) -> None:
    """Print a styled subheader."""
    print(_color(f"▸ {text}", Colors.YELLOW))


def _success(text: str) -> None:
    """Print success message."""
    print(_color(f"✓ {text}", Colors.GREEN))


def _info(text: str) -> None:
    """Print info message."""
    print(_color(f"ℹ {text}", Colors.BLUE))


def _print_steps(steps) -> None:
    """Print step-by-step operations with formatting."""
    for i, (desc, state) in enumerate(steps):
        step_label = _color(f"Step {i + 1}:", Colors.DIM) if i > 0 else ""
        if step_label:
            print(step_label)
        ui.print_matrix(state, _color(desc, Colors.YELLOW))


def _print_bool_result(name: str, value: bool) -> None:
    """Print a boolean result with colored Yes/No."""
    result = _color("Yes", Colors.GREEN) if value else _color("No", Colors.RED)
    print(f"{name}: {result}\n")


# ---------------------------------------------------------------------------
# Operation Handlers
# ---------------------------------------------------------------------------

# -- Basic Arithmetic --

def _handle_addition() -> None:
    _header("Matrix Addition (A + B)")
    _info("Both matrices must have the same dimensions.")
    rows, cols = ui.read_dimensions("A")
    matrix_a = ui.read_matrix(rows, cols, "A")
    matrix_b = ui.read_matrix(rows, cols, "B")
    result = add_matrices(matrix_a, matrix_b)
    _success("Result (A + B):")
    ui.print_matrix(result)


def _handle_subtraction() -> None:
    _header("Matrix Subtraction (A - B)")
    _info("Both matrices must have the same dimensions.")
    rows, cols = ui.read_dimensions("A")
    matrix_a = ui.read_matrix(rows, cols, "A")
    matrix_b = ui.read_matrix(rows, cols, "B")
    result = subtract_matrices(matrix_a, matrix_b)
    _success("Result (A - B):")
    ui.print_matrix(result)


def _handle_scalar_multiplication() -> None:
    _header("Scalar Multiplication (k × A)")
    rows, cols = ui.read_dimensions("A")
    matrix = ui.read_matrix(rows, cols, "A")
    scalar = ui.read_scalar("Enter the scalar k: ")
    result = scalar_multiply(matrix, scalar)
    _success(f"Result ({scalar} × A):")
    ui.print_matrix(result)


def _handle_matrix_multiplication() -> None:
    _header("Matrix Multiplication (A × B)")
    _info("The number of columns in A must equal the number of rows in B.")
    rows_a, cols_a = ui.read_dimensions("A")
    matrix_a = ui.read_matrix(rows_a, cols_a, "A")
    raw_cols_b = input(
        f"Enter the number of columns in matrix B (rows fixed at {cols_a}): "
    ).strip()
    try:
        cols_b = int(raw_cols_b)
    except ValueError as exc:
        raise ValueError("The number of columns must be an integer.") from exc
    if cols_b <= 0:
        raise ValueError("Matrix dimensions must be positive.")
    matrix_b = ui.read_matrix(cols_a, cols_b, "B")
    result = multiply_matrices(matrix_a, matrix_b)
    _success("Result (A × B):")
    ui.print_matrix(result)


def _handle_hadamard() -> None:
    _header("Hadamard Product (A ⊙ B)")
    _info("Element-wise multiplication. Matrices must have the same dimensions.")
    rows, cols = ui.read_dimensions("A")
    matrix_a = ui.read_matrix(rows, cols, "A")
    matrix_b = ui.read_matrix(rows, cols, "B")
    result = hadamard_product(matrix_a, matrix_b)
    _success("Result (A ⊙ B):")
    ui.print_matrix(result)


def _handle_power() -> None:
    _header("Matrix Power (A^n)")
    _info("The matrix must be square. Use n=-1 for the inverse, n=0 for the identity.")
    matrix = ui.read_square_matrix(label="A")
    raw_exp = input("Enter the exponent n (integer >= -1): ").strip()
    try:
        exponent = int(raw_exp)
    except ValueError as exc:
        raise ValueError("The exponent must be an integer.") from exc
    result = matrix_power(matrix, exponent)
    _success(f"Result (A^{exponent}):")
    ui.print_matrix(result)


# -- Transpose and Structure --

def _handle_transpose() -> None:
    _header("Transpose (A^T)")
    matrix = ui.read_matrix(label="A")
    result = transpose(matrix)
    _success("Transpose of A:")
    ui.print_matrix(result)


def _handle_trace() -> None:
    _header("Trace")
    _info("The trace is the sum of the elements on the main diagonal.")
    matrix = ui.read_square_matrix(label="A")
    tr = trace(matrix)
    diag = diagonal(matrix)
    print(f"Main diagonal: {[ui._format_fraction(v) for v in diag]}")
    _success(f"tr(A) = {tr}\n")


def _handle_diagonal() -> None:
    _header("Main Diagonal")
    matrix = ui.read_matrix(label="A")
    diag = diagonal(matrix)
    _success("Main diagonal:")
    print(f"  {[ui._format_fraction(v) for v in diag]}\n")


# -- Determinant and Inverse --

def _handle_determinant() -> None:
    _header("Determinant")
    matrix = ui.read_square_matrix(label="A")
    det = determinant(matrix)
    _success(f"det(A) = {det}\n")


def _handle_inverse() -> None:
    _header("Inverse (with Cofactors and Adjugate)")
    _info("The matrix must be square and non-singular (det ≠ 0).")
    matrix = ui.read_square_matrix(label="A")
    det = determinant(matrix)
    print(f"det(A) = {det}\n")
    if det == 0:
        print(_color("The matrix does not have an inverse (determinant = 0).\n", Colors.RED))
        return
    cof = cofactor_matrix(matrix)
    adj = adjugate(matrix)
    inv = inverse(matrix)
    _subheader("Cofactor Matrix:")
    ui.print_matrix(cof)
    _subheader("Adjugate Matrix (Transpose of the Cofactor Matrix):")
    ui.print_matrix(adj)
    _success("Inverse Matrix A^(-1):")
    ui.print_matrix(inv)


# -- Row Reduction --

def _handle_rref() -> None:
    _header("Reduced Row Echelon Form (RREF)")
    _info("Gauss-Jordan elimination with detailed steps.")
    matrix = ui.read_matrix(label="A")
    _, steps = rref_with_steps(matrix)
    _print_steps(steps)


# -- Rank and Properties --

def _handle_rank() -> None:
    _header("Rank and Nullity")
    _info("Rank = number of non-zero rows in row-echelon form.")
    matrix = ui.read_matrix(label="A")
    r = rank(matrix)
    n = nullity(matrix)
    rows, cols = len(matrix), len(matrix[0])
    print(f"Dimensions: {rows} × {cols}")
    _success(f"Rank: {r}")
    print(f"Nullity: {n}")
    print(f"Check: rank + nullity = {r} + {n} = {r + n} (= number of columns)\n")


def _handle_properties() -> None:
    _header("Matrix Properties")
    matrix = ui.read_square_matrix(label="A")
    print()
    _print_bool_result("Symmetric (A = A^T)", is_symmetric(matrix))
    _print_bool_result("Diagonal", is_diagonal(matrix))
    _print_bool_result("Identity", is_identity(matrix))
    _print_bool_result("Zero matrix (all zeros)", is_zero(matrix))
    _print_bool_result("Upper triangular", is_upper_triangular(matrix))
    _print_bool_result("Lower triangular", is_lower_triangular(matrix))
    det = determinant(matrix)
    print(f"Determinant: {det}")
    _print_bool_result("Invertible (det ≠ 0)", det != 0)


# -- LU Decomposition --

def _handle_lu() -> None:
    _header("LU Decomposition")
    _info("A = L × U, where L is lower triangular and U is upper triangular.")
    _info("This implementation does not use pivoting.")
    matrix = ui.read_square_matrix(label="A")
    L, U, steps = lu_decomposition_with_steps(matrix)
    _print_steps(steps)
    print(_color("Check: L × U =", Colors.YELLOW))
    ui.print_matrix(multiply_matrices(L, U))


# -- Linear Systems --

def _handle_solve_system() -> None:
    _header("Solve Linear System (Ax = b)")
    _info("Find x such that Ax = b using Gaussian elimination.")
    rows_a, cols_a = ui.read_dimensions("A (coefficient matrix)")
    matrix_a = ui.read_matrix(rows_a, cols_a, "A")
    print("Enter vector b (one column):")
    b = ui.read_matrix(rows_a, 1, "b")
    solution, steps = solve_system_with_steps(matrix_a, b)
    _print_steps(steps)
    _success("Solution x:")
    for i, row in enumerate(solution):
        print(f"  x{i + 1} = {ui._format_fraction(row[0])}")
    print()


# -- Generators --

def _handle_create_identity() -> None:
    _header("Generate Identity Matrix")
    raw = input("Enter the order n: ").strip()
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError("The order must be an integer.") from exc
    result = identity_matrix(n)
    _success(f"Identity matrix I_{n}:")
    ui.print_matrix(result)


def _handle_create_zero() -> None:
    _header("Generate Zero Matrix")
    raw = input("Enter the dimensions (rows columns): ").strip()
    try:
        rows, cols = map(int, raw.split())
    except ValueError as exc:
        raise ValueError("Provide two integers separated by spaces.") from exc
    result = zero_matrix(rows, cols)
    _success(f"Zero matrix {rows}×{cols}:")
    ui.print_matrix(result)


# -- Norms --

def _handle_frobenius() -> None:
    _header("Frobenius Norm")
    _info("||A||_F = sqrt(sum of squares of all elements)")
    matrix = ui.read_matrix(label="A")
    norm_sq = frobenius_norm_squared(matrix)
    print(f"||A||_F² = {norm_sq}")
    # Compute approximate sqrt for display
    import math
    approx = math.sqrt(float(norm_sq))
    _success(f"||A||_F ≈ {approx:.6f}\n")


# ---------------------------------------------------------------------------
# Menu System
# ---------------------------------------------------------------------------

MenuItem = Tuple[str, Callable[[], None]]
MenuCategory = Tuple[str, List[Tuple[str, MenuItem]]]

# Categories with their operations
MENU: List[MenuCategory] = [
    (
        "Basic Operations",
        [
            ("1", ("Add matrices (A + B)", _handle_addition)),
            ("2", ("Subtract matrices (A - B)", _handle_subtraction)),
            ("3", ("Multiply by a scalar (k × A)", _handle_scalar_multiplication)),
            ("4", ("Multiply matrices (A × B)", _handle_matrix_multiplication)),
            ("5", ("Hadamard product (A ⊙ B)", _handle_hadamard)),
            ("6", ("Matrix power (A^n)", _handle_power)),
        ],
    ),
    (
        "Structure and Properties",
        [
            ("7", ("Transpose (A^T)", _handle_transpose)),
            ("8", ("Trace", _handle_trace)),
            ("9", ("Main diagonal", _handle_diagonal)),
            ("10", ("Check properties", _handle_properties)),
        ],
    ),
    (
        "Determinant and Inverse",
        [
            ("11", ("Determinant", _handle_determinant)),
            ("12", ("Inverse (with cofactors and adjugate)", _handle_inverse)),
        ],
    ),
    (
        "Reduction and Decomposition",
        [
            ("13", ("Row reduction (RREF)", _handle_rref)),
            ("14", ("LU decomposition", _handle_lu)),
        ],
    ),
    (
        "Linear Systems and Rank",
        [
            ("15", ("Solve system (Ax = b)", _handle_solve_system)),
            ("16", ("Rank and nullity", _handle_rank)),
        ],
    ),
    (
        "Generators and Norms",
        [
            ("17", ("Generate identity matrix", _handle_create_identity)),
            ("18", ("Generate zero matrix", _handle_create_zero)),
            ("19", ("Frobenius norm", _handle_frobenius)),
        ],
    ),
]

# Build flat lookup for quick access
OPTIONS: Dict[str, MenuItem] = {}
for _, items in MENU:
    for key, item in items:
        OPTIONS[key] = item


def _print_menu() -> None:
    """Print the categorized menu."""
    print()
    print(_color("╔══════════════════════════════════════════════════════╗", Colors.CYAN))
    print(_color("║", Colors.CYAN) + _color("              MATRIX TOOLKIT - Main Menu              ", Colors.BOLD) + _color("║", Colors.CYAN))
    print(_color("╚══════════════════════════════════════════════════════╝", Colors.CYAN))
    print()

    for category_name, items in MENU:
        print(_color(f"┌─ {category_name} ", Colors.MAGENTA) + _color("─" * (50 - len(category_name)), Colors.DIM))
        for key, (label, _) in items:
            key_str = _color(f"[{key:>2}]", Colors.GREEN)
            print(f"  {key_str} {label}")
        print()

    print(_color("─" * 56, Colors.DIM))
    exit_str = _color("[ 0]", Colors.RED)
    help_str = _color("[ ?]", Colors.YELLOW)
    print(f"  {exit_str} Exit")
    print(f"  {help_str} Help")
    print()


def _print_help() -> None:
    """Print help information."""
    _header("Help - Matrix Toolkit")
    print("""
This program performs linear algebra operations with exact arithmetic
using fractions. There are no external dependencies beyond the standard library.

""" + _color("How to enter matrices:", Colors.YELLOW) + """
  • Enter the dimensions as "rows columns" (for example: 3 3)
  • For each row, enter the values separated by spaces
  • Fractions can be entered as "1/2", "-3/4", etc.
  • Integers are accepted normally

""" + _color("Input examples:", Colors.YELLOW) + """
  Row: 1 2 3      → [1, 2, 3]
  Row: 1/2 -3 0   → [1/2, -3, 0]

""" + _color("Available operations:", Colors.YELLOW) + """
  • Basic arithmetic: addition, subtraction, multiplication, power
  • Analysis: determinant, inverse, rank, trace
  • Decomposition: RREF, LU
  • Systems: solving Ax = b
  • Properties: symmetry, triangularity, etc.

Press Enter to return to the menu...
""")
    input()


def main() -> None:
    """Main CLI loop."""
    while True:
        _print_menu()
        choice = input(_color("Choose an option: ", Colors.CYAN)).strip()

        if choice == "0":
            print(_color("\n👋 See you soon!\n", Colors.CYAN))
            raise SystemExit

        if choice == "?":
            _print_help()
            continue

        action = OPTIONS.get(choice)
        if not action:
            print(_color("\n⚠ Invalid option. Please try again.\n", Colors.RED))
            continue

        try:
            action[1]()
            input(_color("Press Enter to continue...", Colors.DIM))
        except KeyboardInterrupt:
            print(_color("\n\n⚠ Interrupted by the user.\n", Colors.YELLOW))
        except Exception as exc:
            print(_color(f"\n✗ Error: {exc}\n", Colors.RED))


if __name__ == "__main__":
    main()
