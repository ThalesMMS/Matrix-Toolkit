#
# matriz_fera.py
# Matrix Toolkit
#
# Runs the RREF routine and prints each elimination step, letting users follow
# the Gauss–Jordan process interactively.
#
# Thales Matheus Mendonça Santos - November 2025

"""Reduced row echelon form (RREF) with printed steps."""

from matrix_toolkit import rref_with_steps
from matrix_toolkit import interactive as ui


def main() -> None:
    matriz = ui.read_matrix(label="A")
    # rref_with_steps returns both the result and a log of each transformation.
    _, passos = rref_with_steps(matriz)
    for descricao, estado in passos:
        ui.print_matrix(estado, descricao)


if __name__ == "__main__":
    main()
