"""Matrix Toolkit - Reduced Row Echelon Form helper."""

from fractions import Fraction


def print_matrix(matrix, desc=None):
    """Print the matrix row by row."""
    if desc:
        print(desc)
    for row in matrix:
        formatted = []
        for value in row:
            if isinstance(value, Fraction):
                if value.denominator == 1:
                    formatted.append(f"{value.numerator}")
                else:
                    formatted.append(f"{value.numerator}/{value.denominator}")
            else:
                formatted.append(str(value))
        print("  ".join(formatted))
    print()


def read_matrix():
    """Read a matrix from stdin and return it as Fractions."""
    rows, cols = map(int, input("Enter number of rows and columns (m n): ").split())
    matrix = []
    for index in range(rows):
        row_values = list(map(int, input(f"Enter row {index + 1}: ").split()))
        if len(row_values) != cols:
            raise ValueError("Invalid number of columns.")
        matrix.append([Fraction(value, 1) for value in row_values])
    return matrix


def rref_with_steps(matrix):
    """Convert the matrix to RREF using Gauss–Jordan elimination with printed steps."""
    row_count = len(matrix)
    col_count = len(matrix[0]) if row_count > 0 else 0
    pivot_row = 0

    print_matrix(matrix, "Initial matrix:")

    for col in range(col_count):
        # 1) Find pivot: earliest row with non-zero entry in this column.
        selected = None
        for row in range(pivot_row, row_count):
            if matrix[row][col] != 0:
                selected = row
                break
        if selected is None:
            continue

        # 2) Swap the row into position if needed.
        if selected != pivot_row:
            matrix[pivot_row], matrix[selected] = matrix[selected], matrix[pivot_row]
            print(f"R{pivot_row + 1} <-> R{selected + 1}")
            print_matrix(matrix, f"After swapping R{pivot_row + 1} and R{selected + 1}:")

        # 3) Scale the pivot row so the pivot equals 1.
        pivot_value = matrix[pivot_row][col]
        if pivot_value != 1:
            factor = Fraction(1, 1) / pivot_value
            matrix[pivot_row] = [value * factor for value in matrix[pivot_row]]
            print(f"R{pivot_row + 1} = ({factor.numerator}/{factor.denominator}) * R{pivot_row + 1}")
            print_matrix(matrix, f"After scaling R{pivot_row + 1} to get 1 in column {col + 1}:")

        # 4) Eliminate the column in all other rows.
        for row in range(row_count):
            if row != pivot_row and matrix[row][col] != 0:
                factor = matrix[row][col]
                matrix[row] = [a - factor * b for a, b in zip(matrix[row], matrix[pivot_row])]
                print(f"R{row + 1} = R{row + 1} - ({factor.numerator}/{factor.denominator}) * R{pivot_row + 1}")
                print_matrix(matrix, f"After eliminating entry in R{row + 1}, column {col + 1}:")

        pivot_row += 1
        if pivot_row == row_count:
            break

    print_matrix(matrix, "Matrix in Reduced Row Echelon Form (RREF):")
    return matrix


def main():
    user_matrix = read_matrix()
    rref_with_steps(user_matrix)


if __name__ == "__main__":
    main()
