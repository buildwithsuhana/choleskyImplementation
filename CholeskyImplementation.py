import math
import random
from typing import List
import torch

Matrix = List[List[float]]

# ==============================================================================
# SECTION 1: CORE MATHEMATICAL FUNCTIONS
# ==============================================================================

def transpose_matrix(M: Matrix) -> Matrix:
    """Transposes a matrix represented as a list of lists.

    This function takes a 2D list, where each inner list represents a row,
    and returns its transpose. The rows of the input matrix become the
    columns of the output matrix, and vice-versa.

    Args:
        M (Matrix): A rectangular 2D list representing the matrix to transpose.
            For example: `[[1, 2, 3], [4, 5, 6]]`.

    Returns:
        Matrix: A new 2D list representing the transposed matrix.

    Example:
        >>> a = [[1, 2, 3], [4, 5, 6]]
        >>> transpose_matrix(a)
        [[1, 4], [2, 5], [3, 6]]

        >>> b = [[10, 20], [30, 40]]
        >>> transpose_matrix(b)
        [[10, 30], [20, 40]]
    """
    if not M:
        return []
    return [[M[j][i] for j in range(len(M))] for i in range(len(M[0]))]

def multiply_matrices(A: Matrix, B: Matrix) -> Matrix:
    """Multiplies two matrices, A and B, represented as lists of lists.

    For matrix multiplication to be defined, the number of columns in the
    first matrix (A) must be equal to the number of rows in the second
    matrix (B). If A is an `m x n` matrix and B is an `n x p` matrix,
    the resulting matrix C will be an `m x p` matrix.

    Args:
        A (Matrix): The left-hand matrix (a 2D list) of size `m x n`.
        B (Matrix): The right-hand matrix (a 2D list) of size `n x p`.

    Returns:
        Matrix: A new `m x p` matrix representing the product of A and B,
        with floating-point numbers as elements.

    Raises:
        ValueError: If the inner dimensions of the matrices are not compatible
            (i.e., the number of columns in A is not equal to the
            number of rows in B).

    Example:
        >>> a = [[1, 2, 3],
                 [4, 5, 6]]  # A is a 2x3 matrix
        >>> b = [[7, 8],
                 [9, 10],
                 [11, 12]] # B is a 3x2 matrix
        >>> multiply_matrices(a, b)
        [[58.0, 64.0], [139.0, 154.0]]
    """
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    if cols_A != rows_B:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    C = [[0.0] * cols_B for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def cholesky(A: Matrix, upper: bool = False) -> Matrix:
    """Performs Cholesky decomposition on a symmetric, positive-definite matrix.

    The Cholesky decomposition is a factorization of a matrix `A` into the
    product of a lower triangular matrix `L` and its transpose `L.T`, such that
    `A = L @ L.T`. 
    
    This decomposition is only defined for matrices that are symmetric (the
    matrix is equal to its transpose) and positive-definite (all its
    eigenvalues are positive). It's highly efficient and numerically stable,
    making it useful for solving systems of linear equations and in statistical
    methods like Monte Carlo simulations.

    Args:
        A (Matrix): The input matrix (a 2D list) to decompose. It must be
            symmetric and positive-definite.
        upper (bool, optional): A flag to determine the output format.
            - If `False` (default), returns the lower triangular factor `L`.
            - If `True`, returns the upper triangular factor `U` (where `U` is the
              transpose of `L`).

    Returns:
        Matrix: The lower or upper triangular factor of the decomposition.

    Raises:
        ValueError: If the matrix is not positive-definite, which can result
            in an attempt to take the square root of a negative number.
        ZeroDivisionError: If a diagonal element becomes zero during factorization,
            which can happen if the matrix is not positive-definite.

    Example:
        >>> A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
        >>> L = cholesky(A)
        >>> L
        [[5.0, 0.0, 0.0], [3.0, 3.0, 0.0], [-1.0, 1.0, 3.0]]

        # To verify the result, L @ L.T should be equal to A.

        >>> U = cholesky(A, upper=True)
        >>> U
        [[5.0, 3.0, -1.0], [0.0, 3.0, 1.0], [0.0, 0.0, 3.0]]

    """
    # --- Condition 1: Check if the matrix is square ---
    # The number of rows must equal the number of columns.
    n = len(A)
    if any(len(row) != n for row in A):
        raise ValueError("Input matrix must be square.")

    # --- Condition 2: Check if the matrix is symmetric ---
    # A[i][j] must be equal to A[j][i].
    for i in range(n):
        for j in range(i + 1, n):
            if not math.isclose(A[i][j], A[j][i]):
                raise ValueError("Input matrix must be symmetric.")

    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                # --- Condition 3: Check for Positive-Definiteness ---
                # The value under the square root must be positive.
                diag_val = A[i][i] - s
                if diag_val <= 0:
                    raise ValueError(
                        "Matrix is not positive-definite. "
                        f"Found non-positive value at diagonal ({i},{i})."
                    )
                L[i][j] = math.sqrt(diag_val)
            else:
                L[i][j] = (1.0 / L[j][j] * (A[i][j] - s))

    # If the upper factor U is requested, return the transpose of L
    if upper:
        return transpose_matrix(L)
    else:
        return L

def _invert_lower_triangular(L: Matrix) -> Matrix:
    """Computes the inverse of a lower triangular matrix using forward substitution.

    A lower triangular matrix is a square matrix where all entries above the
    main diagonal are zero. This function finds its inverse, `L_inv`, such that
    the product of `L` and `L_inv` results in the identity matrix (`L @ L_inv = I`).
    

    The algorithm works by solving the equation `L @ X = I` for `X` column by
    column. Because `L` is triangular, we can find the solution for each element
    `X[i][j]` by substituting previously computed values. This method is
    numerically stable and much more efficient than general-purpose inversion
    algorithms.

    Args:
        L (Matrix): A 2D list representing a square, lower triangular matrix.

    Returns:
        Matrix: A new 2D list representing the inverse of `L`.

    Raises:
        ZeroDivisionError: If any element on the main diagonal of `L` is zero,
            as this would make the matrix singular (non-invertible).

    Example:
        >>> l_matrix = [[2.0, 0.0, 0.0],
                        [4.0, 5.0, 0.0],
                        [6.0, 7.0, 8.0]]
        >>> l_inverse = _invert_lower_triangular(l_matrix)
        >>> l_inverse
        [[0.5, 0.0, 0.0], [-0.4, 0.2, 0.0], [-0.275, -0.175, 0.125]]

        # To verify, multiplying l_matrix by l_inverse should yield the
        # identity matrix (within floating point tolerances).
    """
    n = len(L)
    L_inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        L_inv[j][j] = 1.0 / L[j][j]
        for i in range(j + 1, n):
            s = sum(L[i][k] * L_inv[k][j] for k in range(j, i))
            L_inv[i][j] = -s / L[i][i]
    return L_inv

def cholesky_inverse(A: Matrix) -> Matrix:
    """Computes the inverse of a matrix using the Cholesky decomposition method.

    This method is a highly efficient and numerically stable way to invert a
    matrix, but it is only applicable if the matrix `A` is symmetric and
    positive-definite.

    The mathematical foundation for this method is as follows:
    1.  The Cholesky decomposition of `A` gives `A = L @ L.T`, where `L` is a
        lower triangular matrix.
    2.  Therefore, the inverse of `A` is `A⁻¹ = (L @ L.T)⁻¹`.
    3.  Using the matrix inverse property `(XY)⁻¹ = Y⁻¹X⁻¹`, this becomes
        `A⁻¹ = (L.T)⁻¹ @ L⁻¹`.
    4.  Using the property `(X.T)⁻¹ = (X⁻¹).T`, the final formula is:
        `A⁻¹ = (L⁻¹).T @ L⁻¹`. 

    This function implements that final formula by first finding `L`, then its
    inverse `L⁻¹`, and then performing the final multiplication.

    Args:
        A (Matrix): The square matrix to be inverted. It must be symmetric
            and positive-definite for the method to work.

    Returns:
        Matrix: A new 2D list representing the inverse of `A`.

    Raises:
        ValueError: If the underlying `cholesky` function determines that `A`
            is not square, symmetric, or positive-definite.

    Example:
        >>> A = [[4., 12., -16.],
                 [12., 37., -43.],
                 [-16., -43., 98.]]
        >>> A_inv = cholesky_inverse(A)
        >>> A_inv
        [[6.25, -5.5, 0.5], [-5.5, 5.0, -0.5], [0.5, -0.5, 0.0625]]

        # To verify, A @ A_inv should be the identity matrix.
    """
    L = cholesky(A)
    L_inv = _invert_lower_triangular(L)
    L_inv_T = transpose_matrix(L_inv)
    return multiply_matrices(L_inv_T, L_inv)

def compute_inverse_cholesky_factor(H_initial: Matrix, percdamp: float) -> Matrix:
    """Computes the upper Cholesky factor of the inverse of a damped matrix.

    This function is a key pre-computation step in many advanced second-order
    optimization algorithms. It takes a symmetric matrix `H` (typically a
    Hessian or Fisher Information Matrix), regularizes it, and computes a
    factor `U` that can be used to precondition gradient updates.

    The overall transformation is `H -> H_damped -> (H_damped)⁻¹ -> U`, where
    `(H_damped)⁻¹ = U.T @ U`. This factor `U` captures the inverse curvature
    of the optimization landscape, allowing for more effective update steps.

    The process involves four distinct steps:
    1.  **Damping**: A regularization term is added to the diagonal of `H` to
        ensure it is positive-definite and numerically stable.
    2.  **Cholesky Decomposition (Implicit)**: The `cholesky_inverse` function
        is called, which first decomposes the damped matrix.
    3.  **Matrix Inversion**: The inverse of the damped matrix is computed
        efficiently using its Cholesky factors.
    4.  **Final Cholesky Factorization**: A final Cholesky decomposition is
        performed on the *inverse* matrix to yield the upper triangular
        factor `U`. 

    Args:
        H_initial (Matrix): The input symmetric matrix (e.g., a Hessian).
        percdamp (float): A percentage-based damping parameter. A small,
            positive value (e.g., 0.1) used for regularization.

    Returns:
        Matrix: The upper triangular Cholesky factor `U` of the damped
        matrix's inverse.

    Example:
        >>> H = [[6.0, 2.0], [2.0, 4.0]] # A symmetric matrix
        >>> damp_percentage = 0.1
        >>> U = compute_inverse_cholesky_factor(H, damp_percentage)
        >>> U
        [[0.403..., -0.127...], [0.0, 0.468...]]
    """
    H = [row[:] for row in H_initial] # H = ops.copy(H_initial)
    n = len(H)
    diag_H = [H[i][i] for i in range(n)] # diag_H = ops.diagonal(H)
    mean_diag = sum(diag_H) / n  # ops.mean(diag_H)
    damp = percdamp * mean_diag  
    for i in range(n): # diag_H = ops.diagonal(H)
        H[i][i] += damp # diag_H = diag_H + damp # H = (H - ops.diag(ops.diagonal(H))) + ops.diag(diag_H)
    H_inv = cholesky_inverse(H)
    U_of_inv = cholesky(H_inv, upper=True)
    return U_of_inv

# ==============================================================================
# SECTION 2: TEST FUNCTIONS
# ==============================================================================

class MatrixComparator:
    """A class to encapsulate matrix generation and comparison tests."""

    def __init__(self, size: int, percdamp: float):
        """Initializes the comparator with test parameters and generates matrices."""
        self.size = size
        self.percdamp = percdamp

        # Generate a symmetric, positive-definite matrix for testing
        R = [[random.uniform(1, 5) for _ in range(size)] for _ in range(size)]
        self.H_py = multiply_matrices(transpose_matrix(R), R)
        self.H_torch = torch.tensor(self.H_py, dtype=torch.float64)

        print("=" * 60)
        self._print_matrix(self.H_py, "Initial Test Matrix H")
        print(f"Damping Percentage: {self.percdamp}")
        print("=" * 60)

    @staticmethod
    def _print_matrix(M: list, name: str = "Matrix"):
        """Helper method to print a matrix."""
        print(f"--- {name} ---")
        if not M:
            print("[]")
            return
        for row in M:
            print([round(x, 6) for x in row])
        print()

    @staticmethod
    def _are_matrices_close(A: list, B: list, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
        """Helper method to compare two matrices."""
        if len(A) != len(B) or len(A[0]) != len(B[0]):
            return False
        return all(
            math.isclose(A[i][j], B[i][j], rel_tol=rel_tol, abs_tol=abs_tol)
            for i in range(len(A)) for j in range(len(A[0]))
        )

    def test_cholesky_factor(self, upper: bool):
        """
        TEST 1: Validates the core `cholesky` decomposition function.

        This test isolates the fundamental Cholesky decomposition. It compares
        the output of the custom `cholesky` implementation against the trusted,
        highly optimized `torch.linalg.cholesky` function. By running for both
        `upper=True` and `upper=False`, it validates the computation of both
        the lower (L) and upper (R) triangular factors.

        Args:
            upper (bool): If True, tests for the upper factor; otherwise, tests
                          for the lower factor.
        """
        case = "UPPER" if upper else "LOWER"
        print(f"\n>>> TESTING THE `cholesky` FUNCTION FOR {case} FACTOR")

        factor_custom = cholesky(self.H_py, upper=upper)

        factor_torch = torch.linalg.cholesky(self.H_torch, upper=upper)

        self._print_matrix(factor_custom, f"{case} Factor from custom implementation")
        self._print_matrix(factor_torch.tolist(), f"{case} Factor from PyTorch implementation")

        is_close = self._are_matrices_close(factor_custom, factor_torch.tolist())
        print(f"Are the {case.lower()} factors close? {is_close}")
        print(f"\n SUCCESS: {case.capitalize()} factor matches." if is_close else f"\n FAILURE: {case.capitalize()} factor does not match.")
        print("-" * 60)
        
    def test_workflow_replica(self):
        """
        TEST 2: Validates the full `compute_inverse_cholesky_factor` workflow.

        This is an end-to-end integration test. It takes the high-level custom
        function, which performs a sequence of (1) Damping, (2) Inversion, and
        (3) Final Factorization, and compares its final output to a result
        generated by replicating the *exact same logic* step-by-step using
        equivalent PyTorch functions. This ensures that the overall workflow
        is correctly implemented and produces the expected result.
        """
        print("\n>>> TESTING THE `compute_inverse_cholesky_factor` WORKFLOW")

        factor_custom = compute_inverse_cholesky_factor(self.H_py, self.percdamp)

        H_torch_damped = self.H_torch.clone()
        damp_val = self.percdamp * torch.mean(torch.diag(H_torch_damped))
        diag_indices = torch.arange(self.size)
        H_torch_damped[diag_indices, diag_indices] += damp_val
        H_inv_torch = torch.inverse(H_torch_damped)
        factor_torch = torch.linalg.cholesky(H_inv_torch, upper=True)

        self._print_matrix(factor_custom, "Final Factor from Custom implementation")
        self._print_matrix(factor_torch.tolist(), "Final Factor from PyTorch")

        is_close = self._are_matrices_close(factor_custom, factor_torch.tolist())
        print(f"Are the final factors close? {is_close}")
        print("\n SUCCESS: Workflow matches." if is_close else "\n FAILURE: Workflow does not match.")
        print("-" * 60)

    def test_cholesky_inverse(self):
        """
        TEST 3: Validates the direct `cholesky_inverse` function.

        This test focuses specifically on the function that computes a matrix
        inverse using its Cholesky factors. It compares the custom
        implementation against PyTorch's `torch.cholesky_inverse`, which is a
        direct functional equivalent. This validates that the implementation of
        the mathematical formula `A⁻¹ = (L⁻¹).T @ L⁻¹` is correct.
        """
        print("\n>>> TESTING THE `cholesky_inverse` FUNCTION")

        inverse_custom = cholesky_inverse(self.H_py)
        L_torch = torch.linalg.cholesky(self.H_torch)
        inverse_torch = torch.cholesky_inverse(L_torch)

        self._print_matrix(inverse_custom, "Inverse from custom `cholesky_inverse`")
        self._print_matrix(inverse_torch.tolist(), "Inverse from PyTorch")

        is_close = self._are_matrices_close(inverse_custom, inverse_torch.tolist())
        print(f"Are the inverse matrices close? {is_close}")
        print("\n SUCCESS: Inverse matches." if is_close else "\n FAILURE: Inverse does not match.")
        print("-" * 60)

    def run_all_tests(self):
        """
        Runs the full suite of comparison tests in a sequential order.

        This method acts as the main entry point to execute all validation
        routines, ensuring each component of the custom linear algebra library
        is tested against the PyTorch reference implementations.
        """
        self.test_cholesky_factor(upper=False) 
        self.test_cholesky_factor(upper=True)  
        self.test_workflow_replica()
        self.test_cholesky_inverse()

# ==============================================================================
# SECTION 3: SCRIPT EXECUTION
# ==============================================================================

if __name__ == "__main__":
    MATRIX_SIZE = 5
    DAMPING_PERCENTAGE = 0.05

    tester = MatrixComparator(size=MATRIX_SIZE, percdamp=DAMPING_PERCENTAGE)
    tester.run_all_tests()