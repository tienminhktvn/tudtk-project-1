import sys
from pathlib import Path

import numpy as np

# Ensure imports work regardless of current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from part1.gaussian import back_substitution, gaussian_eliminate
from part2.decomposition import qr_decomposition


def solve_gauss_custom(A, b):
    """Giải bằng Khử Gauss (Của Phần 1)"""
    A_list = A.tolist()
    b_list = b.tolist()

    M, x, swaps = gaussian_eliminate(A_list, b_list)
    if isinstance(x, str):
        raise ValueError(f"Hàm Gauss thất bại: {x}")

    return np.array(x, dtype=float)


def solve_qr_custom(A, b):
    """Giải bằng Phân rã QR (Phần 2) + Thế ngược (Phần 1)"""
    Q, R = qr_decomposition(A.tolist())

    # Ép kiểu Q về NumPy array để dùng được .T và np.dot
    Q_arr = np.array(Q, dtype=float)
    y = np.dot(Q_arr.T, b)

    x = back_substitution(R.tolist(), y.tolist())

    if isinstance(x, str):
        raise ValueError(f"Thế ngược thất bại: {x}")

    return np.array(x, dtype=float)


def is_diagonally_dominant(A):
    """Kiểm tra điều kiện hội tụ: Ma trận chéo trội hàng"""
    n = len(A)
    for i in range(n):
        sum_row = sum(abs(A[i][j]) for j in range(n) if i != j)
        if abs(A[i][i]) <= sum_row:
            return False
    return True


def gauss_seidel(A, b, epsilon=1e-6, max_iterations=1000):
    """Thuật toán lặp Gauss-Seidel"""
    n = len(A)
    x = [0.0] * n

    if not is_diagonally_dominant(A):
        pass

    for k in range(max_iterations):
        x_old = list(x)

        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            if A[i][i] == 0:
                raise ValueError("Phần tử trên đường chéo chính bằng 0.")
            x[i] = (b[i] - sum1 - sum2) / A[i][i]

        # Kiểm tra sai số để dừng sớm
        diff = max(abs(x[i] - x_old[i]) for i in range(n))
        if diff < epsilon:
            return x, k + 1

    return x, max_iterations


def solve_gauss_seidel_custom(A, b):
    """Hàm bọc gọi Gauss-Seidel"""
    A_list = A.tolist()
    b_list = b.tolist()
    x, iters = gauss_seidel(A_list, b_list)
    return np.array(x, dtype=float)
