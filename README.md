# Matrix Toolkit

Matrix Toolkit offers classroom-friendly matrix calculations that keep arithmetic exact through Python's `fractions.Fraction`. The core logic lives in a reusable package, and the interactive command line interface is in Portuguese, while all documentation stays in English.

## Features

### Basic Operations
- **Addition, subtraction, scalar multiplication** – Standard matrix arithmetic
- **Matrix multiplication** – Standard product A × B
- **Hadamard product** – Element-wise multiplication A ⊙ B
- **Matrix power** – Raise matrices to integer powers (including A⁻¹)
- **Transpose** – Flip rows and columns

### Determinant and Inverse
- **Determinant** – Recursive Laplace expansion
- **Cofactor matrix** – Matrix of cofactors
- **Adjugate** – Transpose of cofactor matrix
- **Inverse** – Via adjugate method with singularity checks
- **Minor and cofactor** – Individual (i,j) minor and cofactor

### Matrix Reduction
- **RREF** – Reduced Row Echelon Form with step-by-step log
- **LU Decomposition** – Factor A = L × U (without pivoting)

### Linear Systems
- **Solve Ax = b** – Gaussian elimination with step-by-step display
- **Rank** – Number of linearly independent rows/columns
- **Nullity** – Dimension of null space (cols - rank)

### Matrix Properties
- **Trace** – Sum of diagonal elements
- **Diagonal extraction** – Get main diagonal as list
- Property checks: **symmetric**, **diagonal**, **identity**, **zero**, **upper/lower triangular**

### Matrix Generators
- **Identity matrix** – Create I_n
- **Zero matrix** – Create m×n zero matrix

### Norms
- **Frobenius norm** – √(sum of squared entries)

### Additional Utilities
- **Submatrix extraction** – Extract rectangular regions
- **Concatenation** – Horizontal and vertical matrix joining

### Core Features
- Exact arithmetic via `fractions.Fraction`; no third-party dependencies
- Interactive CLI in Portuguese with colored output and categorized menu
- Step-by-step display for educational purposes

## Requirements

- Python 3.8 or newer; standard library only.

## Usage (CLI in Portuguese)

Run the unified menu-driven CLI:

```bash
python scripts/matrix_cli.py
# or
python -m matrix_toolkit.cli
```

The CLI features:
- **Categorized menu** with 19 operations organized into 6 groups
- **Colored output** (ANSI) for better readability (auto-disabled if not a TTY)
- **Step-by-step display** for RREF, LU decomposition, and system solving
- **Help system** with input examples and operation descriptions
- **Fraction support** – Enter values like `1/2`, `-3/4` directly

Available categories:
1. **Operacoes Basicas** – Sum, subtract, scalar multiply, matrix multiply, Hadamard, power
2. **Estrutura e Propriedades** – Transpose, trace, diagonal, property checks
3. **Determinante e Inversa** – Determinant, inverse with cofactors
4. **Reducao e Decomposicao** – RREF, LU decomposition
5. **Sistemas Lineares e Posto** – Solve Ax=b, rank/nullity
6. **Geradores e Normas** – Identity, zero matrix, Frobenius norm

Legacy-style entry points remain for single tasks:
- `python scripts/matriz_fera.py` – RREF with printed row operations.
- `python scripts/matriz_inversa_cofatores.py` – Determinant, cofactors, adjugate, and inverse.

## Using the library in Python code

```python
from matrix_toolkit import (
    add_matrices, determinant, inverse, trace, rank,
    identity_matrix, is_symmetric, lu_decomposition,
    solve_system, matrix_power, rref
)

a = [[1, 2], [3, 4]]
b = [[2, 0], [1, 2]]

# Basic operations
sum_ab = add_matrices(a, b)
det_a = determinant(a)
inv_a = inverse(a)  # raises ValueError if singular

# New operations
tr = trace(a)                    # Sum of diagonal: 1 + 4 = 5
r = rank(a)                      # Matrix rank
I3 = identity_matrix(3)          # 3x3 identity
sym = is_symmetric(a)            # False
L, U = lu_decomposition(a)       # A = L × U
a_squared = matrix_power(a, 2)   # A²
reduced = rref(a)                # Row echelon form

# Solve linear system Ax = b
A = [[2, 1], [1, 3]]
b = [[5], [10]]
x = solve_system(A, b)           # Solution vector
```

All functions accept any numeric input coercible to `Fraction` and perform validation (shape checks, square requirements, etc.).

## Repository Structure

```
README.md
LICENSE
matrix_toolkit/
  __init__.py          # Package exports
  cli.py               # Portuguese CLI with colored menu
  interactive.py       # Input/output helpers (Portuguese prompts)
  operations.py        # Pure matrix operations built on Fraction
scripts/
  matrix_cli.py                 # Entry point for the CLI menu
  matriz_fera.py                # RREF demo script
  matriz_inversa_cofatores.py   # Inverse demo script
```

## License

This project is distributed under the MIT License. See `LICENSE` for details.
