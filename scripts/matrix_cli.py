#
# matrix_cli.py
# Matrix Toolkit
#
# Simple launcher that delegates to the packaged CLI entry point so the toolkit
# can be invoked as a script without dealing with Python module paths.
#
# Thales Matheus Mendonça Santos - November 2025

"""Ponto de entrada para o CLI do Matrix Toolkit."""

from matrix_toolkit.cli import main


if __name__ == "__main__":
    # Keep this script tiny; it simply hands control to the packaged CLI.
    main()
