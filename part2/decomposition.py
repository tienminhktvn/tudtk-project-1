import numpy as np 
import math as m

#tich vo huong hai vecto 
def dot(u, v):
    s = 0
    for i in range(len(u)):
        s += u[i]*v[i]
        
    return s 

#hàm hỗ trợ tính độ dài vector v
def norm(v):
    return m.sqrt(dot(v, v))
    
# hàm hỗ trợ tính vector trừ trong bài
def projection(u, v):
    coef = dot(u,v)/dot(v,v)
    return [coef*x for x in v]
    
def qr_decomposition(A):
    A = np.array(A, dtype = float)#biến ma trận A về mảng
    n, m = A.shape #n là số dòng, m là số cột

    Q = [] #ma trận Q gồm những vector qi
    R = [[0]*m for _ in range(m)]  #ma trận vector toàn 0
    
    for j in range (m):
        v = A[:, j].tolist() #chuyển cột về ma trận về
        for i in range(j):
            q_i = Q[i] #lấy cái nhỏ trước đó ra
            R[i][j] = dot(q_i, A[:,j].tolist())# nhân với u mỗi cột rả về phân tử của R
            proj = projection(A[:, j].tolist(), q_i) #
            v = [v[k] - proj[k] for k in range(len(v))]
        R[j][j] = norm(v)
        #nếu R[j,j] == 0 thì tức là v_j lúc này = 0 kết thuc bài toán
        if R[j][j] <= 1e-10:
            raise ValueError(f"Thuật toán kết thúc tại {j+1} vì vector v_{j+1} = 0!!")
        
        #q_j = [x / R[j,j] for x in v]
        
    
        q_j = [x / R[j][j] for x in v]
        Q.append(q_j)
        
    Q = np.array(Q).T
    R = np.array(R)
    
    return Q, R
    
import numpy as np

def verify_qr(A, Q, R, tol=1e-8):
    A = np.array(A, dtype=float)
    print("=== VERIFY QR ===")

    

    #1. Kiểm tra A = QR
    A_new = np.dot(Q, R)
    err1 = np.max(np.abs(A - A_new))

    print("1. A ≈ QR:")
    print("   max error =", err1, "->", "OK" if err1 < tol else "FAIL")

    #2. Kiểm tra Q trực chuẩn: Q^T Q = I
    QtQ = np.dot(Q.T, Q)
    I = np.eye(QtQ.shape[0])#tạo ma trân đợn vị kích thước bằng với ma trận Q

    err2 = np.max(np.abs(QtQ - I))

    print("2. Q trực chuẩn (Q^T Q = I):")
    print("   max error =", err2, "->", "OK" if err2 < tol else "FAIL")

    #3. Kiểm tra R tam giác trên
    lower_part = np.tril(R, -1)  # lấy phần dưới đường chéo
    err3 = np.max(np.abs(lower_part))

    print("3. R tam giác trên:")
    print("   max lower =", err3, "->", "OK" if err3 < tol else "FAIL")

    #4. So sánh với NumPy (chỉ tham khảo)
    Q_np, R_np = np.linalg.qr(A)

    # lưu ý: Q có thể khác dấu → so sánh trị tuyệt đối
    err4 = np.max(np.abs(np.abs(Q) - np.abs(Q_np)))

    print("4. So với NumPy:")
    print("   max diff =", err4, "->", "OK" if err4 < tol else "FAIL")

    print("\n")
    
    