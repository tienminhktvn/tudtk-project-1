import time
import numpy as np
from scipy.linalg import hilbert

def generate_random_system(n):
    """Dùng cho: Đo Hiệu Năng Thời Gian (Vẽ đồ thị O(n^3))"""
    A = np.random.rand(n, n)
    x_true = np.random.rand(n)
    b = A @ x_true
    return A, b

def generate_hilbert_system(n):
    """Dùng cho: Phân Tích Ổn Định Số Học (Ma trận Ill-conditioned - Số điều kiện cực lớn)"""
    A = hilbert(n)
    x_true = np.random.rand(n)
    b = A @ x_true
    return A, b


def generate_spd_system(n):
    """Dùng cho: Phân Tích Ổn Định Số Học (Ma trận Well-conditioned - Rất ổn định)"""
    M = np.random.rand(n, n)
    A = M @ M.T + np.eye(n) * n
    x_true = np.random.rand(n)
    b = A @ x_true
    return A, b


def generate_diagonally_dominant_system(n):
    """Dùng cho: Test riêng thuật toán Gauss-Seidel (Ép ma trận thành chéo trội)"""
    A = np.random.rand(n, n)
    for i in range(n):
        A[i, i] += np.sum(np.abs(A[i, :])) + 1.0
    x_true = np.random.rand(n)
    b = A @ x_true
    return A, b

def evaluate_solver(solver_func, A, b, num_runs=5):
    """
    Nhiệm vụ: Chạy thuật toán num_runs lần.
    Trả về: Thời gian chạy trung bình, Sai số tương đối
    """
    times = []
    x_pred = None

    # Chạy 5 lần để lấy thời gian trung bình
    for _ in range(num_runs):
        start_time = time.perf_counter()
        x_pred = solver_func(A, b)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    avg_time = np.mean(times)

    # Tính sai số tương đối: ||Ax_pred - b||_2 / ||b||_2
    residual_vector = A @ x_pred - b
    residual_norm = np.linalg.norm(residual_vector, ord=2)
    b_norm = np.linalg.norm(b, ord=2)
    relative_error = residual_norm / b_norm if b_norm != 0 else residual_norm
    return avg_time, relative_error