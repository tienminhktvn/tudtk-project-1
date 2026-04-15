def inverse(A):
    """
    Tìm ma trận nghịch đảo A^-1 bằng phương pháp Gauss-Jordan.
    """
    n = len(A)
    # Kiểm tra ma trận vuông
    if any(len(row) != n for row in A):
        return "Lỗi: Ma trận phải là ma trận vuông."

    # Tính epsilon động
    max_val_A = max((abs(A[i][j]) for i in range(n) for j in range(n)), default=0)
    epsilon = max(1e-12, max_val_A * 1e-12)

    # Tạo ma trận ghép [A | I]
    M = [A[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    # 1. Khử xuôi (Gauss elimination)
    for k in range(n):
        # Partial Pivoting
        max_idx = k
        for i in range(k + 1, n):
            if abs(M[i][k]) > abs(M[max_idx][k]):
                max_idx = i
        
        if abs(M[max_idx][k]) < epsilon:
            return "Lỗi: Ma trận suy biến, không có ma trận nghịch đảo."
        
        M[k], M[max_idx] = M[max_idx], M[k]

        # Chuẩn hóa hàng k sao cho pivot = 1
        pivot = M[k][k]
        for j in range(k, 2 * n):
            M[k][j] /= pivot

        # Khử các hàng còn lại (cả trên và dưới hàng k)
        for i in range(n):
            if i != k:
                factor = M[i][k]
                for j in range(k, 2 * n):
                    M[i][j] -= factor * M[k][j]

    # Trích xuất ma trận nghịch đảo từ nửa bên phải của M
    inv_A = [row[n:] for row in M]
    return inv_A