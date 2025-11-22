#
# interactive.py
# Matrix Toolkit
#
# Input/output helpers for the CLI, handling matrix sizes, entries, and display
# formatting while validating user-provided fractions for reliable operations.
#
# Thales Matheus Mendonça Santos - November 2025

"""CLI helpers for interactive matrix input/output (Portuguese prompts)."""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

Matrix = List[List[Fraction]]


def _parse_fraction(value: str) -> Fraction:
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Entrada invalida: {value}") from exc


def read_dimensions(label: str | None = None) -> Tuple[int, int]:
    target = f" da matriz {label}" if label else ""
    raw = input(f"Digite o numero de linhas e colunas{target} (m n): ").strip()
    try:
        rows, cols = map(int, raw.split())
    except ValueError as exc:
        raise ValueError("Forneca dois inteiros separados por espaco.") from exc
    if rows <= 0 or cols <= 0:
        raise ValueError("As dimensoes devem ser positivas.")
    return rows, cols


def read_matrix(rows: int | None = None, cols: int | None = None, label: str | None = None) -> Matrix:
    if rows is None or cols is None:
        rows, cols = read_dimensions(label)

    target = f" da matriz {label}" if label else ""
    matrix: Matrix = []
    for r in range(rows):
        line = input(
            f"Digite a linha {r + 1}{target} (use espacos entre valores, ex: 1 -2 3/4): "
        ).strip()
        entries = line.split()  # Split by whitespace to allow fractions like 3/4.
        if len(entries) != cols:
            raise ValueError("Numero incorreto de colunas para esta linha.")
        matrix.append([_parse_fraction(value) for value in entries])
    print()
    return matrix


def read_square_matrix(order: int | None = None, label: str | None = None) -> Matrix:
    if order is None:
        raw = input(f"Digite a ordem da matriz{f' {label}' if label else ''} (n): ").strip()
        try:
            order = int(raw)
        except ValueError as exc:
            raise ValueError("A ordem deve ser um inteiro.") from exc
    if order <= 0:
        raise ValueError("A ordem deve ser positiva.")
    # Reuse read_matrix to centralize parsing/validation.
    return read_matrix(order, order, label)


def read_scalar(prompt: str = "Digite o escalar: ") -> Fraction:
    raw = input(prompt).strip()
    return _parse_fraction(raw)


def _format_fraction(value: Fraction) -> str:
    # Display integers plainly and proper fractions with a slash.
    return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(
        value.numerator
    )


def format_matrix(matrix: Sequence[Sequence[Fraction]]) -> str:
    lines = []
    for row in matrix:
        formatted = "  ".join(_format_fraction(value) for value in row)
        lines.append(f"[ {formatted} ]")
    return "\n".join(lines)


def print_matrix(matrix: Sequence[Sequence[Fraction]], heading: str | None = None) -> None:
    if heading:
        print(heading)
    print(format_matrix(matrix))
    print()
