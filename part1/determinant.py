def determinant(A):
    """
    Tính định thức của ma trận vuông A bằng phương pháp khử Gauss.
    """
    n = len(A)
    # Kiểm tra ma trận vuông
    if any(len(row) != n for row in A):
        raise ValueError("Ma trận phải là ma trận vuông để tính định thức.")
        
    # Sao chép ma trận
    M = [row[:] for row in A]
    swaps = 0
    
    # Tính epsilon động dựa trên phần tử lớn nhất của ma trận A
    max_val_A = max((abs(A[i][j]) for i in range(n) for j in range(n)), default=0)
    epsilon = max(1e-12, max_val_A * 1e-12)
    
    for k in range(n):
        # Partial pivoting
        max_val = 0
        max_idx = k
        for i in range(k, n):
            if abs(M[i][k]) > max_val:
                max_val = abs(M[i][k])
                max_idx = i
                
        # Nếu pivot = 0, ma trận suy biến, định thức bằng 0
        if max_val < epsilon:
            return 0.0
            
        # Hoán đổi dòng
        if max_idx != k:
            M[k], M[max_idx] = M[max_idx], M[k]
            swaps += 1
            
        # Khử dòng dưới
        for i in range(k + 1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n):
                M[i][j] -= factor * M[k][j]
                
    # Tính tích các phần tử trên đường chéo chính
    det = (-1) ** swaps
    for i in range(n):
        det *= M[i][i]
        
    return det