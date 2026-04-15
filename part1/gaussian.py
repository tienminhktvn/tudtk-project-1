def back_substitution(U, c):
    """
    Giải hệ phương trình tam giác trên Ux = c.
    """
    if not U or not U[0]:
        return "Ma trận rỗng."
        
    m = len(U)
    n = len(U[0])
    
    #Tính epsilon động dựa trên phần tử lớn nhất của U
    max_val = max((abs(U[i][j]) for i in range(m) for j in range(n)), default=0)
    epsilon = max(1e-12, max_val * 1e-12)
    
    pivot_cols = []
    pivot_rows = []
    
    # 1. Tìm pivot và kiểm tra tính tương thích 
    for i in range(m):
        pivot_found = False
        for j in range(n):
            if abs(U[i][j]) > epsilon:
                pivot_cols.append(j)
                pivot_rows.append(i)
                pivot_found = True
                break
                
        # Nếu dòng không có pivot (toàn số 0) nhưng c[i] khác 0 -> Vô nghiệm
        if not pivot_found and abs(c[i]) > epsilon:
            return "Hệ vô nghiệm."
            
    #Xác định nghiệm duy nhất dựa vào số lượng cột pivot
    free_cols = [j for j in range(n) if j not in pivot_cols]
    
    # 2. Đưa về RREF (Rút gọn) để giải quyết các mâu thuẫn mapping giữa pivot và free variables
    U_rref = [row[:] for row in U]
    c_rref = c[:]
    
    for i in reversed(range(len(pivot_cols))):
        p_col = pivot_cols[i]
        p_row = pivot_rows[i]
        
        # Chuẩn hóa pivot về 1
        pivot_val = U_rref[p_row][p_col]
        for j in range(p_col, n):
            U_rref[p_row][j] /= pivot_val
        c_rref[p_row] /= pivot_val
        
        # Khử các phần tử phía trên pivot
        for k in range(p_row):
            factor = U_rref[k][p_col]
            for j in range(p_col, n):
                U_rref[k][j] -= factor * U_rref[p_row][j]
            c_rref[k] -= factor * c_rref[p_row]

    # 3. Xuất kết quả
    if len(pivot_cols) == n:
        # Nghiệm duy nhất 
        x = [0.0] * n
        for i in range(n):
            x[pivot_cols[i]] = c_rref[pivot_rows[i]]
        return x
        
    else:
        # Vô số nghiệm
        print("hệ không có nghiệm duy nhất")
        param_map = {j: f"t{idx + 1}" for idx, j in enumerate(free_cols)}
        solution_items = []
        
        for i in range(len(pivot_cols)):
            p_col = pivot_cols[i]
            p_row = pivot_rows[i]
            
            val = c_rref[p_row]
            terms = []
            
            if abs(val) > epsilon:
                terms.append(f"{val:.4g}")
                
            for j in free_cols:
                coef = -U_rref[p_row][j]
                if abs(coef) > epsilon:
                    #Nối chuỗi dấu an toàn hơn
                    sign = "+" if coef > 0 else "-"
                    coef_abs = abs(coef)
                    coef_str = f"{coef_abs:.4g}" if abs(coef_abs - 1.0) > epsilon else ""
                    terms.append(f"{sign} {coef_str}{param_map[j]}")
            
            expr = " ".join(terms) if terms else "0"
            if expr.startswith("+ "): 
                expr = expr[2:]
            elif expr.startswith("- "): 
                expr = "-" + expr[2:]
                
            solution_items.append((p_col, f"x_{p_col} = {expr}"))
            
        for j in free_cols:
            solution_items.append((j, f"x_{j} = {param_map[j]} (với {param_map[j]} thuộc R)"))
            
        solution_items.sort(key=lambda item: item[0])
        return "\n".join(item[1] for item in solution_items)


def gaussian_eliminate(A, b):
    """
    Phép khử Gauss có Partial Pivoting.
    """
    # Xử lý ma trận rỗng
    if not A or not A[0]:
        return [], "Ma trận rỗng.", 0
        
    m = len(A)
    n = len(A[0])
    
    # Epsilon động cho việc khử Gauss
    max_val_A = max((abs(A[i][j]) for i in range(m) for j in range(n)), default=0)
    epsilon = max(1e-12, max_val_A * 1e-12)
    
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    swaps = 0
    row = 0
    col = 0
    
    while row < m and col < n:
        max_val = 0
        max_idx = row
        for i in range(row, m):
            if abs(M[i][col]) > max_val:
                max_val = abs(M[i][col])
                max_idx = i
                
        if max_val < epsilon:
            print(f"không có pivot tại cột {col}")
            col += 1
            continue
            
        if max_idx != row:
            M[row], M[max_idx] = M[max_idx], M[row]
            swaps += 1
            
        for i in range(row + 1, m):
            factor = M[i][col] / M[row][col]
            for j in range(col, n + 1):
                M[i][j] -= factor * M[row][j]
                
        row += 1
        col += 1
        
    U = [r[:n] for r in M]
    c = [r[n] for r in M]
    
    x = back_substitution(U, c)
    return M, x, swaps

