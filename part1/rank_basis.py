def rank_and_basis(A):
    """
    Tính hạng và tìm cơ sở của không gian cột, dòng, và nghiệm.
    """
    if not A or not A[0]:
        return "Ma trận rỗng."

    m, n = len(A), len(A[0])
    max_val_A = max((abs(A[i][j]) for i in range(m) for j in range(n)), default=0)
    epsilon = max(1e-12, max_val_A * 1e-12)

    # Bước 1: Đưa ma trận về RREF (Reduced Row Echelon Form)
    M = [row[:] for row in A]
    pivot_cols = []
    row = 0
    for col in range(n):
        if row >= m: break
        # Tìm pivot
        max_idx = row
        for i in range(row + 1, m):
            if abs(M[i][col]) > abs(M[max_idx][col]):
                max_idx = i
        
        if abs(M[max_idx][col]) < epsilon:
            continue # Cột này không có pivot, bỏ qua sang cột tiếp theo
        
        M[row], M[max_idx] = M[max_idx], M[row]
        pivot_cols.append(col)
        
        # Chuẩn hóa pivot = 1
        pivot_val = M[row][col]
        for j in range(col, n):
            M[row][j] /= pivot_val
            
        # Khử tất cả các hàng khác (Gauss-Jordan)
        for i in range(m):
            if i != row:
                factor = M[i][col]
                for j in range(col, n):
                    M[i][j] -= factor * M[row][j]
        row += 1

    rank = len(pivot_cols)

    # Bước 2: Tìm cơ sở các không gian
    # Cơ sở không gian dòng (Row Space): Các dòng khác 0 trong RREF
    basis_row = [M[i] for i in range(rank)]

    # Cơ sở không gian cột (Column Space): Các cột gốc tại vị trí pivot
    basis_col = []
    for c in pivot_cols:
        col_vector = [A[r][c] for r in range(m)]
        basis_col.append(col_vector)

    # Cơ sở không gian nghiệm (Null Space): Giải Ax = 0
    basis_null = []
    free_cols = [j for j in range(n) if j not in pivot_cols]
    
    for f_col in free_cols:
        null_vec = [0.0] * n
        null_vec[f_col] = 1.0 # Gán ẩn tự do = 1
        for i, p_col in enumerate(pivot_cols):
            # Từ RREF: x_pivot + sum(coef * x_free) = 0 => x_pivot = -sum(coef * x_free)
            null_vec[p_col] = -M[i][f_col]
        basis_null.append(null_vec)

    return {
        "rank": rank,
        "basis_row": basis_row,
        "basis_col": basis_col,
        "basis_null": basis_null
    }