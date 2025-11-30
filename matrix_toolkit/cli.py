#
# cli.py
# Matrix Toolkit
#
# Text-based menu for performing common matrix operations interactively,
# delegating computation to the core library and showing intermediate steps.
#
# Thales Matheus Mendonça Santos - November 2025

"""Interactive CLI (Portuguese) for common matrix operations."""

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
        step_label = _color(f"Passo {i + 1}:", Colors.DIM) if i > 0 else ""
        if step_label:
            print(step_label)
        ui.print_matrix(state, _color(desc, Colors.YELLOW))


def _print_bool_result(name: str, value: bool) -> None:
    """Print a boolean result with colored Yes/No."""
    result = _color("Sim", Colors.GREEN) if value else _color("Nao", Colors.RED)
    print(f"{name}: {result}\n")


# ---------------------------------------------------------------------------
# Operation Handlers
# ---------------------------------------------------------------------------

# -- Basic Arithmetic --

def _handle_addition() -> None:
    _header("Soma de Matrizes (A + B)")
    _info("As duas matrizes devem ter as mesmas dimensoes.")
    rows, cols = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows, cols, "A")
    matriz_b = ui.read_matrix(rows, cols, "B")
    resultado = add_matrices(matriz_a, matriz_b)
    _success("Resultado (A + B):")
    ui.print_matrix(resultado)


def _handle_subtraction() -> None:
    _header("Subtracao de Matrizes (A - B)")
    _info("As duas matrizes devem ter as mesmas dimensoes.")
    rows, cols = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows, cols, "A")
    matriz_b = ui.read_matrix(rows, cols, "B")
    resultado = subtract_matrices(matriz_a, matriz_b)
    _success("Resultado (A - B):")
    ui.print_matrix(resultado)


def _handle_scalar_multiplication() -> None:
    _header("Multiplicacao por Escalar (k × A)")
    rows, cols = ui.read_dimensions("A")
    matriz = ui.read_matrix(rows, cols, "A")
    escalar = ui.read_scalar("Digite o escalar k: ")
    resultado = scalar_multiply(matriz, escalar)
    _success(f"Resultado ({escalar} × A):")
    ui.print_matrix(resultado)


def _handle_matrix_multiplication() -> None:
    _header("Multiplicacao de Matrizes (A × B)")
    _info("Numero de colunas de A deve ser igual ao numero de linhas de B.")
    rows_a, cols_a = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows_a, cols_a, "A")
    raw_cols_b = input(
        f"Digite o numero de colunas da matriz B (linhas fixas = {cols_a}): "
    ).strip()
    try:
        cols_b = int(raw_cols_b)
    except ValueError as exc:
        raise ValueError("O numero de colunas deve ser um inteiro.") from exc
    if cols_b <= 0:
        raise ValueError("As dimensoes da matriz devem ser positivas.")
    matriz_b = ui.read_matrix(cols_a, cols_b, "B")
    resultado = multiply_matrices(matriz_a, matriz_b)
    _success("Resultado (A × B):")
    ui.print_matrix(resultado)


def _handle_hadamard() -> None:
    _header("Produto de Hadamard (A ⊙ B)")
    _info("Multiplicacao elemento a elemento. As matrizes devem ter mesmas dimensoes.")
    rows, cols = ui.read_dimensions("A")
    matriz_a = ui.read_matrix(rows, cols, "A")
    matriz_b = ui.read_matrix(rows, cols, "B")
    resultado = hadamard_product(matriz_a, matriz_b)
    _success("Resultado (A ⊙ B):")
    ui.print_matrix(resultado)


def _handle_power() -> None:
    _header("Potencia de Matriz (A^n)")
    _info("A matriz deve ser quadrada. Use n=-1 para inversa, n=0 para identidade.")
    matriz = ui.read_square_matrix(label="A")
    raw_exp = input("Digite o expoente n (inteiro >= -1): ").strip()
    try:
        exponent = int(raw_exp)
    except ValueError as exc:
        raise ValueError("O expoente deve ser um inteiro.") from exc
    resultado = matrix_power(matriz, exponent)
    _success(f"Resultado (A^{exponent}):")
    ui.print_matrix(resultado)


# -- Transpose and Structure --

def _handle_transpose() -> None:
    _header("Transposta (A^T)")
    matriz = ui.read_matrix(label="A")
    resultado = transpose(matriz)
    _success("Transposta de A:")
    ui.print_matrix(resultado)


def _handle_trace() -> None:
    _header("Traco (Trace)")
    _info("O traco eh a soma dos elementos da diagonal principal.")
    matriz = ui.read_square_matrix(label="A")
    tr = trace(matriz)
    diag = diagonal(matriz)
    print(f"Diagonal principal: {[ui._format_fraction(v) for v in diag]}")
    _success(f"tr(A) = {tr}\n")


def _handle_diagonal() -> None:
    _header("Diagonal Principal")
    matriz = ui.read_matrix(label="A")
    diag = diagonal(matriz)
    _success("Diagonal principal:")
    print(f"  {[ui._format_fraction(v) for v in diag]}\n")


# -- Determinant and Inverse --

def _handle_determinant() -> None:
    _header("Determinante")
    matriz = ui.read_square_matrix(label="A")
    det = determinant(matriz)
    _success(f"det(A) = {det}\n")


def _handle_inverse() -> None:
    _header("Inversa (com Cofatores e Adjunta)")
    _info("A matriz deve ser quadrada e nao-singular (det ≠ 0).")
    matriz = ui.read_square_matrix(label="A")
    det = determinant(matriz)
    print(f"det(A) = {det}\n")
    if det == 0:
        print(_color("A matriz nao possui inversa (determinante = 0).\n", Colors.RED))
        return
    cof = cofactor_matrix(matriz)
    adj = adjugate(matriz)
    inv = inverse(matriz)
    _subheader("Matriz de Cofatores:")
    ui.print_matrix(cof)
    _subheader("Matriz Adjunta (Transposta dos Cofatores):")
    ui.print_matrix(adj)
    _success("Matriz Inversa A^(-1):")
    ui.print_matrix(inv)


# -- Row Reduction --

def _handle_rref() -> None:
    _header("Forma Escalonada Reduzida (RREF)")
    _info("Gauss-Jordan elimination com passos detalhados.")
    matriz = ui.read_matrix(label="A")
    _, passos = rref_with_steps(matriz)
    _print_steps(passos)


# -- Rank and Properties --

def _handle_rank() -> None:
    _header("Posto (Rank) e Nulidade")
    _info("Posto = numero de linhas nao-nulas na forma escalonada.")
    matriz = ui.read_matrix(label="A")
    r = rank(matriz)
    n = nullity(matriz)
    rows, cols = len(matriz), len(matriz[0])
    print(f"Dimensoes: {rows} × {cols}")
    _success(f"Posto (rank): {r}")
    print(f"Nulidade (nullity): {n}")
    print(f"Verificacao: rank + nullity = {r} + {n} = {r + n} (= numero de colunas)\n")


def _handle_properties() -> None:
    _header("Propriedades da Matriz")
    matriz = ui.read_square_matrix(label="A")
    print()
    _print_bool_result("Simetrica (A = A^T)", is_symmetric(matriz))
    _print_bool_result("Diagonal", is_diagonal(matriz))
    _print_bool_result("Identidade", is_identity(matriz))
    _print_bool_result("Matriz nula (todos zeros)", is_zero(matriz))
    _print_bool_result("Triangular superior", is_upper_triangular(matriz))
    _print_bool_result("Triangular inferior", is_lower_triangular(matriz))
    det = determinant(matriz)
    print(f"Determinante: {det}")
    _print_bool_result("Invertivel (det ≠ 0)", det != 0)


# -- LU Decomposition --

def _handle_lu() -> None:
    _header("Decomposicao LU")
    _info("A = L × U, onde L eh triangular inferior e U eh triangular superior.")
    _info("Esta implementacao nao usa pivoteamento.")
    matriz = ui.read_square_matrix(label="A")
    L, U, passos = lu_decomposition_with_steps(matriz)
    _print_steps(passos)
    print(_color("Verificacao: L × U =", Colors.YELLOW))
    ui.print_matrix(multiply_matrices(L, U))


# -- Linear Systems --

def _handle_solve_system() -> None:
    _header("Resolver Sistema Linear (Ax = b)")
    _info("Encontra x tal que Ax = b usando eliminacao de Gauss.")
    rows_a, cols_a = ui.read_dimensions("A (matriz de coeficientes)")
    matriz_a = ui.read_matrix(rows_a, cols_a, "A")
    print("Digite o vetor b (uma coluna):")
    b = ui.read_matrix(rows_a, 1, "b")
    solution, passos = solve_system_with_steps(matriz_a, b)
    _print_steps(passos)
    _success("Solucao x:")
    for i, row in enumerate(solution):
        print(f"  x{i + 1} = {ui._format_fraction(row[0])}")
    print()


# -- Generators --

def _handle_create_identity() -> None:
    _header("Gerar Matriz Identidade")
    raw = input("Digite a ordem n: ").strip()
    try:
        n = int(raw)
    except ValueError as exc:
        raise ValueError("A ordem deve ser um inteiro.") from exc
    resultado = identity_matrix(n)
    _success(f"Matriz identidade I_{n}:")
    ui.print_matrix(resultado)


def _handle_create_zero() -> None:
    _header("Gerar Matriz Nula")
    raw = input("Digite as dimensoes (linhas colunas): ").strip()
    try:
        rows, cols = map(int, raw.split())
    except ValueError as exc:
        raise ValueError("Forneca dois inteiros separados por espaco.") from exc
    resultado = zero_matrix(rows, cols)
    _success(f"Matriz nula {rows}×{cols}:")
    ui.print_matrix(resultado)


# -- Norms --

def _handle_frobenius() -> None:
    _header("Norma de Frobenius")
    _info("||A||_F = sqrt(soma dos quadrados de todos elementos)")
    matriz = ui.read_matrix(label="A")
    norm_sq = frobenius_norm_squared(matriz)
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
        "Operacoes Basicas",
        [
            ("1", ("Somar matrizes (A + B)", _handle_addition)),
            ("2", ("Subtrair matrizes (A - B)", _handle_subtraction)),
            ("3", ("Multiplicar por escalar (k × A)", _handle_scalar_multiplication)),
            ("4", ("Multiplicar matrizes (A × B)", _handle_matrix_multiplication)),
            ("5", ("Produto de Hadamard (A ⊙ B)", _handle_hadamard)),
            ("6", ("Potencia de matriz (A^n)", _handle_power)),
        ],
    ),
    (
        "Estrutura e Propriedades",
        [
            ("7", ("Transposta (A^T)", _handle_transpose)),
            ("8", ("Traco (trace)", _handle_trace)),
            ("9", ("Diagonal principal", _handle_diagonal)),
            ("10", ("Verificar propriedades", _handle_properties)),
        ],
    ),
    (
        "Determinante e Inversa",
        [
            ("11", ("Determinante", _handle_determinant)),
            ("12", ("Inversa (com cofatores e adjunta)", _handle_inverse)),
        ],
    ),
    (
        "Reducao e Decomposicao",
        [
            ("13", ("Forma escalonada (RREF)", _handle_rref)),
            ("14", ("Decomposicao LU", _handle_lu)),
        ],
    ),
    (
        "Sistemas Lineares e Posto",
        [
            ("15", ("Resolver sistema (Ax = b)", _handle_solve_system)),
            ("16", ("Posto (rank) e nulidade", _handle_rank)),
        ],
    ),
    (
        "Geradores e Normas",
        [
            ("17", ("Gerar matriz identidade", _handle_create_identity)),
            ("18", ("Gerar matriz nula", _handle_create_zero)),
            ("19", ("Norma de Frobenius", _handle_frobenius)),
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
    print(_color("║", Colors.CYAN) + _color("            MATRIX TOOLKIT - Menu Principal            ", Colors.BOLD) + _color("║", Colors.CYAN))
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
    print(f"  {exit_str} Sair")
    print(f"  {help_str} Ajuda")
    print()


def _print_help() -> None:
    """Print help information."""
    _header("Ajuda - Matrix Toolkit")
    print("""
Este programa realiza operacoes de algebra linear com aritmetica exata
usando fracoes. Nao ha dependencias externas alem da biblioteca padrao.

""" + _color("Como inserir matrizes:", Colors.YELLOW) + """
  • Digite as dimensoes como "linhas colunas" (ex: 3 3)
  • Para cada linha, digite os valores separados por espacos
  • Fracoes podem ser digitadas como "1/2", "-3/4", etc.
  • Inteiros sao aceitos normalmente

""" + _color("Exemplos de entrada:", Colors.YELLOW) + """
  Linha: 1 2 3      → [1, 2, 3]
  Linha: 1/2 -3 0   → [1/2, -3, 0]

""" + _color("Operacoes disponiveis:", Colors.YELLOW) + """
  • Aritmetica basica: soma, subtracao, multiplicacao, potencia
  • Analise: determinante, inversa, posto, traco
  • Decomposicao: RREF, LU
  • Sistemas: resolucao de Ax = b
  • Propriedades: simetria, triangularidade, etc.

Pressione Enter para voltar ao menu...
""")
    input()


def main() -> None:
    """Main CLI loop."""
    while True:
        _print_menu()
        choice = input(_color("Escolha uma opcao: ", Colors.CYAN)).strip()

        if choice == "0":
            print(_color("\n👋 Ate logo!\n", Colors.CYAN))
            raise SystemExit

        if choice == "?":
            _print_help()
            continue

        action = OPTIONS.get(choice)
        if not action:
            print(_color("\n⚠ Opcao invalida. Tente novamente.\n", Colors.RED))
            continue

        try:
            action[1]()
            input(_color("Pressione Enter para continuar...", Colors.DIM))
        except KeyboardInterrupt:
            print(_color("\n\n⚠ Interrompido pelo usuario.\n", Colors.YELLOW))
        except Exception as exc:
            print(_color(f"\n✗ Erro: {exc}\n", Colors.RED))


if __name__ == "__main__":
    main()
