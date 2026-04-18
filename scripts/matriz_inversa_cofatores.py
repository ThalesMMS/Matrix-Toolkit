#
# matriz_inversa_cofatores.py
# Matrix Toolkit
#
# Computes determinant, cofactor matrix, adjugate, and inverse for a square
# matrix, printing intermediate results so users can see how A^{-1} is built.
#
# Thales Matheus Mendonça Santos - November 2025

"""Determinant, cofactors, adjugate, and inverse of a square matrix."""

from matrix_toolkit import adjugate, cofactor_matrix, determinant, inverse
from matrix_toolkit import interactive as ui


def main() -> None:
    matriz = ui.read_square_matrix(label="A")
    det = determinant(matriz)
    print(f"det(A) = {det}\n")
    if det == 0:
        print("The matrix is not invertible (zero determinant).")
        return

    # Cofactor and adjugate highlight the structure used to build the inverse.
    cofatores = cofactor_matrix(matriz)
    adj = adjugate(matriz)
    try:
        inv = inverse(matriz)
    except ValueError as exc:
        print(f"Failed to compute the inverse: {exc}")
        return

    ui.print_matrix(cofatores, "Cofactor matrix:")
    ui.print_matrix(adj, "Adjugate matrix:")
    ui.print_matrix(inv, "Inverse matrix A^(-1):")


if __name__ == "__main__":
    main()
