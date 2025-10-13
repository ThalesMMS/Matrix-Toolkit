# Matrix Toolkit

Matrix Toolkit gathers small, self-contained Python scripts that help with classroom linear algebra exercises. Each script lives under `scripts/`, guides the user interactively, and keeps calculations exact by relying on Python's `fractions.Fraction`.

- `scripts/matriz_fera.py` performs Gauss–Jordan elimination step by step, showing every row operation that leads a matrix to its reduced row echelon form (RREF).
- `scripts/matriz_inversa_cofatores.py` computes the determinant, cofactor matrix, adjugate and inverse of a square matrix using the classical cofactor method.

## Requirements

- Python 3.8 or newer.  
  No third-party libraries are necessary; the scripts use only the standard library.

## Usage

Run the scripts directly with Python. They will prompt for the matrix dimensions and entries.

### Reduced Row Echelon Form (RREF)

```bash
python scripts/matriz_fera.py
```

Example session:

```
Digite linhas e colunas (m n): 3 4
Digite a linha 1: 1 2 3 4
Digite a linha 2: 0 1 4 5
Digite a linha 3: 2 3 4 5
```

The program then prints every row operation and the final FER(A) (RREF) matrix.

### Inverse via Cofactors

```bash
python scripts/matriz_inversa_cofatores.py
```

Sample interaction:

```
Digite a ordem da matriz (n): 3
Digite a linha 1 (separada por espaços, ex: 1 2.5 3/4): 1 0 2
Digite a linha 2 (separada por espaços, ex: 1 2.5 3/4): 0 1 0
Digite a linha 3 (separada por espaços, ex: 1 2.5 3/4): 4 0 1
```

If the determinant is zero, the script explains that the matrix has no inverse; otherwise it prints the determinant, cofactor matrix, adjugate and inverse.

## Repository Structure

```
README.md
LICENSE
.gitignore
scripts/
  matriz_fera.py                # RREF with detailed Gauss–Jordan steps
  matriz_inversa_cofatores.py   # Determinant and inverse via cofactors
```

## License

This project is distributed under the MIT License. See `LICENSE` for details.
