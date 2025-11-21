"""Determinante, cofatores, adjunta e inversa de matriz quadrada."""

from matrix_toolkit import adjugate, cofactor_matrix, determinant, inverse
from matrix_toolkit import interactive as ui


def main() -> None:
    matriz = ui.read_square_matrix(label="A")
    det = determinant(matriz)
    print(f"det(A) = {det}\n")
    if det == 0:
        print("A matriz nao e invertivel (determinante zero).")
        return

    cofatores = cofactor_matrix(matriz)
    adj = adjugate(matriz)
    try:
        inv = inverse(matriz)
    except ValueError as exc:
        print(f"Falha ao calcular a inversa: {exc}")
        return

    ui.print_matrix(cofatores, "Matriz de cofatores:")
    ui.print_matrix(adj, "Matriz adjunta:")
    ui.print_matrix(inv, "Matriz inversa A^(-1):")


if __name__ == "__main__":
    main()
