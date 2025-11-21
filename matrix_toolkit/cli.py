"""Interactive CLI (Portuguese) for common matrix operations."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

from . import (
    add_matrices,
    adjugate,
    cofactor_matrix,
    determinant,
    inverse,
    multiply_matrices,
    rref_with_steps,
    scalar_multiply,
    subtract_matrices,
    transpose,
)
from . import interactive as ui


def _header(title: str) -> None:
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def _print_steps(steps) -> None:
    for desc, state in steps:
        ui.print_matrix(state, desc)


def _handle_addition() -> None:
    _header("Soma de matrizes")
    rows, cols = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows, cols, "A")
    matriz_b = ui.read_matrix(rows, cols, "B")
    resultado = add_matrices(matriz_a, matriz_b)
    ui.print_matrix(resultado, "Resultado (A + B):")


def _handle_subtraction() -> None:
    _header("Subtracao de matrizes")
    rows, cols = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows, cols, "A")
    matriz_b = ui.read_matrix(rows, cols, "B")
    resultado = subtract_matrices(matriz_a, matriz_b)
    ui.print_matrix(resultado, "Resultado (A - B):")


def _handle_scalar_multiplication() -> None:
    _header("Multiplicacao por escalar")
    rows, cols = ui.read_dimensions("A")
    matriz = ui.read_matrix(rows, cols, "A")
    escalar = ui.read_scalar("Digite o escalar k: ")
    resultado = scalar_multiply(matriz, escalar)
    ui.print_matrix(resultado, "Resultado (k * A):")


def _handle_matrix_multiplication() -> None:
    _header("Multiplicacao de matrizes")
    rows_a, cols_a = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows_a, cols_a, "A")
    print("Para multiplicar, o numero de colunas de A deve ser igual ao numero de linhas de B.")
    raw_cols_b = input(f"Digite o numero de colunas da matriz B (linhas fixas = {cols_a}): ").strip()
    try:
        cols_b = int(raw_cols_b)
    except ValueError as exc:
        raise ValueError("O numero de colunas deve ser um inteiro.") from exc
    if cols_b <= 0:
        raise ValueError("As dimensoes da matriz devem ser positivas.")

    matriz_b = ui.read_matrix(cols_a, cols_b, "B")
    resultado = multiply_matrices(matriz_a, matriz_b)
    ui.print_matrix(resultado, "Resultado (A * B):")


def _handle_transpose() -> None:
    _header("Transposta")
    matriz = ui.read_matrix(label="A")
    resultado = transpose(matriz)
    ui.print_matrix(resultado, "Transposta de A:")


def _handle_determinant() -> None:
    _header("Determinante")
    matriz = ui.read_square_matrix(label="A")
    det = determinant(matriz)
    print(f"det(A) = {det}\n")


def _handle_inverse() -> None:
    _header("Inversa, cofatores e adjunta")
    matriz = ui.read_square_matrix(label="A")
    det = determinant(matriz)
    print(f"det(A) = {det}\n")
    if det == 0:
        print("A matriz nao possui inversa porque o determinante eh zero.\n")
        return
    cof = cofactor_matrix(matriz)
    adj = adjugate(matriz)
    inv = inverse(matriz)
    ui.print_matrix(cof, "Matriz de cofatores:")
    ui.print_matrix(adj, "Matriz adjunta:")
    ui.print_matrix(inv, "Matriz inversa A^(-1):")


def _handle_rref() -> None:
    _header("Forma escalonada reduzida (RREF)")
    matriz = ui.read_matrix(label="A")
    _, passos = rref_with_steps(matriz)
    _print_steps(passos)


def _exit() -> None:
    print("Ate logo!")
    raise SystemExit


OPTIONS: Dict[str, Tuple[str, Callable[[], None]]] = {
    "1": ("Somar matrizes", _handle_addition),
    "2": ("Subtrair matrizes", _handle_subtraction),
    "3": ("Multiplicar por escalar", _handle_scalar_multiplication),
    "4": ("Multiplicar matrizes", _handle_matrix_multiplication),
    "5": ("Transposta", _handle_transpose),
    "6": ("Determinante", _handle_determinant),
    "7": ("Inversa (com cofatores e adjunta)", _handle_inverse),
    "8": ("Forma escalonada reduzida (RREF)", _handle_rref),
    "0": ("Sair", _exit),
}


def main() -> None:
    while True:
        print("=== Matrix Toolkit ===")
        for key in sorted(OPTIONS.keys()):
            label = OPTIONS[key][0]
            print(f"{key} - {label}")
        choice = input("Escolha uma opcao: ").strip()
        action = OPTIONS.get(choice)
        if not action:
            print("Opcao invalida.\n")
            continue
        try:
            action[1]()
        except KeyboardInterrupt:
            print("\nInterrompido pelo usuario.\n")
        except Exception as exc:
            print(f"Erro: {exc}\n")


if __name__ == "__main__":
    main()
