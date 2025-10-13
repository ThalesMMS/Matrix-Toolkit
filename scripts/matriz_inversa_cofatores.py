from fractions import Fraction


def print_matrix(matrix, desc=None):
    """Print the matrix row by row with optional description."""
    if desc:
        print(desc)
    for row in matrix:
        formatted = []
        for element in row:
            if isinstance(element, Fraction):
                if element.denominator == 1:
                    formatted.append(f"{element.numerator}")
                else:
                    formatted.append(f"{element.numerator}/{element.denominator}")
            else:
                formatted.append(str(element))
        print("[", "  ".join(formatted), "]")
    print()


def read_square_matrix():
    """Read an n x n matrix from user input and return it as Fractions."""
    order = int(input("Enter the order of the matrix (n): "))
    matrix = []
    for row_index in range(order):
        entries = input(
            f"Enter row {row_index + 1} (space separated, e.g.: 1 2.5 3/4): "
        ).split()
        if len(entries) != order:
            raise ValueError("Row does not contain the expected number of columns.")
        row = []
        for entry in entries:
            try:
                row.append(Fraction(entry))
            except ValueError:
                raise ValueError(f"Invalid entry: {entry}")
        matrix.append(row)
    print()
    return matrix


def minor_matrix(matrix, row_to_remove, col_to_remove):
    """Return the minor matrix omitting the provided row and column."""
    size = len(matrix)
    return [
        [matrix[r][c] for c in range(size) if c != col_to_remove]
        for r in range(size)
        if r != row_to_remove
    ]


def determinant(matrix, pivot_row=0):
    """Compute the determinant recursively via Laplace expansion."""
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = Fraction(0, 1)
    for column in range(size):
        sign = Fraction((-1) ** (pivot_row + column), 1)
        submatrix = minor_matrix(matrix, pivot_row, column)
        det += matrix[pivot_row][column] * sign * determinant(submatrix)
    return det


def cofactor_matrix(matrix):
    """Return the matrix of cofactors."""
    size = len(matrix)
    cofactors = []
    for row in range(size):
        cofactor_row = []
        for column in range(size):
            submatrix = minor_matrix(matrix, row, column)
            sign = Fraction((-1) ** (row + column), 1)
            cofactor_row.append(sign * determinant(submatrix))
        cofactors.append(cofactor_row)
    return cofactors


def transpose(matrix):
    """Return the transpose of a matrix."""
    return [list(column) for column in zip(*matrix)]


def inverse_via_cofactors(matrix):
    """Return determinant, cofactor matrix, adjugate, and inverse (if invertible)."""
    det_matrix = determinant(matrix)
    if det_matrix == 0:
        return det_matrix, None, None, None

    cof_matrix = cofactor_matrix(matrix)
    adjugate = transpose(cof_matrix)
    inverse = [
        [element * (Fraction(1, 1) / det_matrix) for element in row]
        for row in adjugate
    ]
    return det_matrix, cof_matrix, adjugate, inverse


def main():
    user_matrix = read_square_matrix()
    print_matrix(user_matrix, "Original matrix:")

    det_matrix, cof_matrix, adjugate, inverse = inverse_via_cofactors(user_matrix)
    print(f"Determinant: {det_matrix}\n")

    if det_matrix == 0:
        print("Matrix is not invertible (determinant = 0).")
        return

    print_matrix(cof_matrix, "Cofactor matrix:")
    print_matrix(adjugate, "Adjugate matrix (transpose of cofactors):")
    print_matrix(inverse, "Inverse matrix A^(-1):")


if __name__ == "__main__":
    main()
