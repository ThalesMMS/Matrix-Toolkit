from fractions import Fraction
import unittest
from unittest.mock import patch

from matrix_toolkit import operations


def _multiply_matrix_vector(matrix, vector):
    return [
        [sum(Fraction(value) * vector[col][0] for col, value in enumerate(row))]
        for row in matrix
    ]


class SolveSystemTests(unittest.TestCase):
    def test_solve_system_unique_solution_satisfies_ax_equals_b_without_calling_rank(self):
        A = [[2, 1], [1, -1]]
        b = [[5], [1]]

        with patch.object(operations, "rank", side_effect=AssertionError("rank called")):
            solution = operations.solve_system(A, b)

        self.assertEqual(solution, [[Fraction(2)], [Fraction(1)]])
        self.assertEqual(_multiply_matrix_vector(A, solution), b)

    def test_solve_system_with_steps_unique_solution_and_step_log_without_calling_rank(self):
        A = [[2, 1], [1, -1]]
        b = [[5], [1]]

        with patch.object(operations, "rank", side_effect=AssertionError("rank called")):
            solution, steps = operations.solve_system_with_steps(A, b)

        self.assertEqual(solution, [[Fraction(2)], [Fraction(1)]])
        self.assertEqual(_multiply_matrix_vector(A, solution), b)
        self.assertTrue(steps)
        self.assertTrue(all(isinstance(step, tuple) and len(step) == 2 for step in steps))
        self.assertEqual(steps[-1], ("Solution x", solution))

    def test_solve_system_raises_for_inconsistent_system(self):
        A = [[1, 1], [2, 2]]
        b = [[3], [7]]

        with self.assertRaisesRegex(ValueError, "The system has no solution"):
            operations.solve_system(A, b)

    def test_solve_system_with_steps_raises_for_inconsistent_system(self):
        A = [[1, 1], [2, 2]]
        b = [[3], [7]]

        with self.assertRaisesRegex(ValueError, "The system has no solution"):
            operations.solve_system_with_steps(A, b)

    def test_solve_system_raises_for_underdetermined_system(self):
        A = [[1, 1, 1], [2, 2, 2]]
        b = [[3], [6]]

        with self.assertRaisesRegex(ValueError, "infinitely many solutions"):
            operations.solve_system(A, b)

    def test_solve_system_with_steps_raises_for_underdetermined_system(self):
        A = [[1, 1, 1], [2, 2, 2]]
        b = [[3], [6]]

        with self.assertRaisesRegex(ValueError, "infinitely many solutions"):
            operations.solve_system_with_steps(A, b)

    def test_solve_system_handles_one_by_one_system(self):
        A = [[4]]
        b = [[10]]

        self.assertEqual(operations.solve_system(A, b), [[Fraction(5, 2)]])

    def test_solve_system_handles_identity_matrix(self):
        A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        b = [[7], [-3], [2]]

        self.assertEqual(operations.solve_system(A, b), [[Fraction(7)], [Fraction(-3)], [Fraction(2)]])

    def test_solve_system_with_steps_handles_already_diagonal_system(self):
        A = [[2, 0, 0], [0, -3, 0], [0, 0, 4]]
        b = [[6], [9], [10]]

        solution, steps = operations.solve_system_with_steps(A, b)

        self.assertEqual(solution, [[Fraction(3)], [Fraction(-3)], [Fraction(5, 2)]])
        self.assertEqual(steps[-1], ("Solution x", solution))


if __name__ == "__main__":
    unittest.main()
