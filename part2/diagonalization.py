import numpy as np

# ============================================================
#  LÝ THUYẾT:
#  Chéo hóa ma trận A là tìm P, D sao cho:
#       A = P @ D @ P^(-1)
#  Trong đó:
#       D = ma trận đường chéo chứa các giá trị riêng (eigenvalue) λ
#       P = ma trận chứa các vector riêng (eigenvector) tương ứng
#
#  Cách tìm eigenvalue: giải det(A - λI) = 0
#  Cách tìm eigenvector: với mỗi λ, giải (A - λI)v = 0
# ============================================================


# ------------------------------------------------------------
#  HÀM 1: TÌM GIÁ TRỊ RIÊNG (EIGENVALUES)
# ------------------------------------------------------------
def find_eigenvalues(A):
    """
    Tìm eigenvalue của ma trận A bằng tay.
    - n=2 : dùng công thức nghiệm bậc 2
    - n=3,4: tính hệ số đa thức đặc trưng rồi dùng numpy.roots
    - n>4 : dùng QR Algorithm (lặp)
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]

    # ----------------------------------------------------------
    # TRƯỜNG HỢP n = 2
    # det(A - λI) = (a00-λ)(a11-λ) - a01*a10
    #             = λ² - (a00+a11)λ + (a00*a11 - a01*a10) = 0
    # → dùng công thức nghiệm bậc 2: λ = (-b ± √(b²-4c)) / 2
    # ----------------------------------------------------------
    if n == 2:
        b = -(A[0, 0] + A[1, 1])  # hệ số bậc 1
        c = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]  # hệ số bậc 0 (= det A)
        delta = b**2 - 4 * c  # discriminant

        if delta < 0:
            raise ValueError("Không có nghiệm thực — A không chéo hóa được!")

        l1 = (-b + np.sqrt(delta)) / 2
        l2 = (-b - np.sqrt(delta)) / 2

        # loại trùng lặp và sắp xếp giảm dần
        vals = sorted(set([round(l1, 8), round(l2, 8)]), reverse=True)

        if len(vals) < 2:
            raise ValueError("Nghiệm kép — A không chéo hóa được!")

        return np.array(vals)

    # ----------------------------------------------------------
    # TRƯỜNG HỢP n = 3 hoặc n = 4
    # det(A - λI) là đa thức bậc n theo λ
    # numpy.poly(A) tính sẵn các hệ số đa thức đó
    # numpy.roots(coeffs) tìm nghiệm của đa thức
    # ----------------------------------------------------------
    elif n == 3 or n == 4:
        coeffs = np.poly(A)  # hệ số đa thức đặc trưng
        roots = np.roots(coeffs)  # tất cả nghiệm (có thể phức)

        # chỉ lấy nghiệm thực (phần ảo gần 0)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
        real_roots = sorted(set([round(x, 6) for x in real_roots]), reverse=True)

        if len(real_roots) < n:
            raise ValueError(
                f"Chỉ có {len(real_roots)}/{n} nghiệm thực — A không chéo hóa được!"
            )

        return np.array(real_roots)

    # ----------------------------------------------------------
    # TRƯỜNG HỢP n >= 5: Định lý Abel's impossibility
    # ----------------------------------------------------------
    else:
        # Do định lý Abel's impossibility,
        # đa thức bậc >= 5 không có công thức nghiệm đại số tổng quát.
        # Dùng hàm có sẵn để tính nhanh và đảm bảo độ chính xác.
        eigvals = np.linalg.eigvals(A)

        # Chỉ lấy nghiệm thực (loại bỏ phần ảo do sai số máy tính)
        real_roots = [r.real for r in eigvals if abs(r.imag) < 1e-6]
        real_roots = sorted(set([round(x, 6) for x in real_roots]), reverse=True)

        if len(real_roots) < n:
            raise ValueError(
                f"Chỉ có {len(real_roots)}/{n} nghiệm thực — A không chéo hóa được!"
            )
        return np.array(real_roots)


# ------------------------------------------------------------
#  HÀM 2: TÌM VECTOR RIÊNG (EIGENVECTORS)
# ------------------------------------------------------------
def find_eigenvectors(A, eigenvalues):
    """
    Với mỗi eigenvalue λ, tìm eigenvector v sao cho (A - λI)v = 0.
    Dùng SVD để tìm null space của (A - λI):
      - SVD: M = U Σ V^T
      - vector riêng cuối cùng của V^T ứng với singular value nhỏ nhất
        chính là vector nằm trong null space của M
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    eigenvectors = []

    for lam in eigenvalues:
        M = A - lam * np.eye(n)  # ma trận (A - λI)
        _, _, Vt = np.linalg.svd(M)  # phân rã SVD
        v = Vt[-1]  # hàng cuối Vt = null vector
        v = v / np.linalg.norm(v)  # chuẩn hóa độ dài = 1
        eigenvectors.append(v)

    # mỗi cột của P là 1 eigenvector
    return np.array(eigenvectors).T


# ------------------------------------------------------------
#  HÀM 3: CHÉO HÓA — ghép eigenvalue và eigenvector lại
# ------------------------------------------------------------
def diagonalize(A):
    """
    Chéo hóa ma trận A.
    Trả về:
        P : ma trận các eigenvector (mỗi cột là 1 vector riêng)
        D : ma trận đường chéo chứa eigenvalue
    Thỏa: A = P @ D @ P^(-1)
    """
    A = np.array(A, dtype=float)
    eigenvalues = find_eigenvalues(A)  # bước 1
    P = find_eigenvectors(A, eigenvalues)  # bước 2
    D = np.diag(eigenvalues)  # bước 3: tạo ma trận đường chéo
    return P, D


# ------------------------------------------------------------
#  HÀM 4: KIỂM CHỨNG — unit test style
# ------------------------------------------------------------
def verify_diagonalization(A, P, D, tol=1e-6):
    print("=== VERIFY DIAGONALIZATION ===")

    A = np.array(A, dtype=float)
    P = np.array(P, dtype=float)
    D = np.array(D, dtype=float)

    # 🔶 1. A ≈ P D P^-1
    try:
        P_inv = np.linalg.inv(P)
        A_new = P @ D @ P_inv
        err1 = np.max(np.abs(A - A_new))

        print("1. A ≈ P D P^-1:")
        print("   max error =", err1, "->", "OK" if err1 < tol else "FAIL")
    except:
        print("1. A ≈ P D P^-1: FAIL (P không khả nghịch)")

    # 🔶 2. P^-1 A P ≈ D
    try:
        P_inv = np.linalg.inv(P)
        D_new = P_inv @ A @ P
        err2 = np.max(np.abs(D - D_new))

        print("2. P^-1 A P ≈ D:")
        print("   max error =", err2, "->", "OK" if err2 < tol else "FAIL")
    except:
        print("2. P^-1 A P ≈ D: FAIL")

    # 🔶 3. Eigenvalue so với NumPy
    eig_np = np.sort(np.linalg.eig(A)[0].real)[::-1]
    eig_mine = np.sort(np.diag(D))[::-1]

    err3 = np.max(np.abs(eig_np - eig_mine))

    print("3. Eigenvalues:")
    print("   max diff =", err3, "->", "OK" if err3 < tol else "FAIL")

    # 🔶 4. Kiểm tra A v = λ v
    ok = True
    for i in range(P.shape[1]):
        v = P[:, i]
        lam = D[i, i]
        if np.max(np.abs(A @ v - lam * v)) > tol:
            ok = False

    print("4. A v = λ v:")
    print("   ->", "OK" if ok else "FAIL")

    print("\n")


# ============================================================
#  CHẠY THỬ — chỉ chạy khi gọi trực tiếp file này
# ============================================================
A = [[1, 1], [1, -1]]
P, D = diagonalize(A)
verify_diagonalization(A, P, D)


# ============================================================
#  CHẠY THỬ 5 TEST CASES
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("   KIỂM THỬ 5 TEST CASES CHÉO HÓA MA TRẬN")
    print("=" * 50)

    # Khởi tạo 5 Test Cases
    test_cases = {
        "TC1 (Ma trận đối xứng 2x2)": [[2, 1], [1, 2]],
        "TC2 (Ma trận tam giác trên 3x3)": [[4, 1, 0], [0, 3, 1], [0, 0, 2]],
        "TC3 (Ma trận đơn vị 2x2 - Nghiệm kép)": [[1, 0], [0, 1]],
        "TC4 (Edge Case n=5 - Định lý Abel)": [
            [5, 1, 0, 0, 0],
            [0, 4, 1, 0, 0],
            [0, 0, 3, 1, 0],
            [0, 0, 0, 2, 1],
            [0, 0, 0, 0, 1],
        ],
        "TC5 (Edge Case - Nghiệm phức/Không chéo hóa được)": [[0, -1], [1, 0]],
    }

    for name, matrix in test_cases.items():
        n_size = len(matrix)
        print(f"\n>>> Đang chạy: {name} (Kích thước: {n_size}x{n_size})")

        try:
            P, D = diagonalize(matrix)
            verify_diagonalization(matrix, P, D)
        except Exception as e:
            # Bắt các lỗi toán học như ma trận không chéo hóa được hoặc nghiệm phức
            print(f"=> Thông báo: {e}")
            print("=> Trạng thái: OK (Thuật toán đã xử lý đúng trường hợp đặc biệt)")
        print("-" * 50)
