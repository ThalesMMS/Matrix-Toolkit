"""Forma escalonada reduzida (RREF) com passos mostrados."""

from matrix_toolkit import rref_with_steps
from matrix_toolkit import interactive as ui


def main() -> None:
    matriz = ui.read_matrix(label="A")
    _, passos = rref_with_steps(matriz)
    for descricao, estado in passos:
        ui.print_matrix(estado, descricao)


if __name__ == "__main__":
    main()
