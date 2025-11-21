# Matrix Toolkit

Matrix Toolkit offers classroom-friendly matrix calculations that keep arithmetic exact through Python's `fractions.Fraction`. The core logic now lives in a reusable package, and the interactive command line interface is in Portuguese, while all documentation stays in English.

## Features

- Addition, subtraction, scalar multiplication, and matrix multiplication.
- Transpose, determinant, cofactor matrix, adjugate, and inverse (with singularity checks).
- Reduced Row Echelon Form (RREF) with a step-by-step log.
- Exact arithmetic via `fractions.Fraction`; no third-party dependencies.
- Interactive CLI in Portuguese for quick calculations.

## Requirements

- Python 3.8 or newer; standard library only.

## Usage (CLI in Portuguese)

Run the unified menu-driven CLI:

```bash
python scripts/matrix_cli.py
# or
python -m matrix_toolkit.cli
```

Available actions:

- Somar, subtrair, multiplicar por escalar, multiplicar matrizes.
- Transposta, determinante, inversa (mostrando cofatores e adjunta).
- Forma escalonada reduzida (RREF) com todos os passos.

Legacy-style entry points remain for single tasks:

- `python scripts/matriz_fera.py` – RREF with printed row operations.
- `python scripts/matriz_inversa_cofatores.py` – Determinant, cofactors, adjugate, and inverse.

## Using the library in Python code

```python
from matrix_toolkit import add_matrices, determinant, inverse

a = [[1, 2], [3, 4]]
b = [[2, 0], [1, 2]]
sum_ab = add_matrices(a, b)
det_a = determinant(a)
inv_a = inverse(a)  # raises ValueError if singular
```

All functions accept any numeric input coercible to `Fraction` and perform validation (shape checks, square requirements, etc.).

## Repository Structure

```
README.md
LICENSE
matrix_toolkit/
  __init__.py
  cli.py                 # Portuguese CLI menu
  interactive.py         # Input/output helpers (Portuguese prompts)
  operations.py          # Pure matrix operations built on Fraction
scripts/
  matrix_cli.py          # Entry point for the CLI menu
  matriz_fera.py         # RREF, uses the shared operations
  matriz_inversa_cofatores.py   # Determinant, cofatores, adjunta, inversa
```

## License

This project is distributed under the MIT License. See `LICENSE` for details.
