# Đồ án 1: Ma Trận và Cơ Sở của Tính Toán Khoa Học

**Môn học:** Toán Ứng Dụng và Thống Kê (MTH00051)

## Thông tin nhóm: Group_8

| STT | Họ và Tên            |   MSSV   | Công việc chính                              |
| :-: | :------------------- | :------: | :------------------------------------------- |
|  1  | Nguyễn Mai Tùng Hiếu | 24120311 | Phần 3: Phân tích hiệu năng, benchmark       |
|  2  | Trần Ngọc Trúc Ly    | 24120376 | Phần 2: Phân rã QR & Trực quan Manim         |
|  3  | Nguyễn Hồ Quang Tiến | 24120463 | Phần 1: Khử Gauss, Giải hệ, Lặp Gauss-Seidel |
|  4  | Vũ Đức Trung         | 24120479 | Phần 1: Khử Gauss, Giải hệ, Lặp Gauss-Seidel |
|  5  | Cao Tiến Minh        | 22120207 | Phần 2: Phân rã QR & Trực quan Manim         |

---

## Project Structure

```text
Group_8/
├── README.md               # Tài liệu hướng dẫn này
├── requirements.txt        # Danh sách các thư viện cần thiết (numpy, scipy, manim, matplotlib,...)
├── report/
│   ├── report.pdf          # Báo cáo chính thức của đồ án (PDF)
│   └── report.tex          # Mã nguồn LaTeX của báo cáo
├── part1/
│   ├── gaussian.py         # Code thuật toán khử Gauss và back-substitution
│   ├── determinant.py      # Code tính định thức
│   ├── inverse.py          # Code tìm ma trận nghịch đảo
│   ├── rank_basis.py       # Code tìm hạng và cơ sở
│   └── part1_demo.ipynb    # Notebook chạy thử 6 Test Cases và kiểm chứng Phần 1
├── part2/
│   ├── decomposition.py    # Code thuật toán phân rã QR (Gram-Schmidt)
│   ├── diagonalization.py  # Code tìm trị riêng, vector riêng và chéo hóa
│   ├── manim_scene.py      # Kịch bản mã nguồn Manim trực quan hóa
│   └── demo_video.mp4      # Video hoạt ảnh kết quả (Thời lượng >= 2 phút)
└── part3/
    ├── solvers.py          # Tập hợp API giải hệ (Gauss, QR, Gauss-Seidel)
    ├── benchmark.py        # Logic đo lường thời gian và sinh ma trận test (Hilbert, SPD)
    └── analysis.ipynb      # Notebook phân tích thực nghiệm, vẽ đồ thị và đánh giá
```

## Hướng dẫn cài đặt

Cài đặt các thư viện thông qua `pip`:

```bash
pip install -r requirements.txt
```

## Chi tiết

### 1. Phần 1: Phép khử Gauss và Ứng dụng

- **Source:** Nằm trong thư mục `part1/`. Các thuật toán được cài đặt hoàn toàn thủ công bằng List của Python, tích hợp epsilon động và cơ chế bắt lỗi ma trận suy biến.
- **Notebook**: File part1/part1_demo.ipynb và bấm Run All. Notebook sẽ tự động gọi 6 Test Cases, tính toán nghiệm và dùng thư viện NumPy để đối chiếu sai số $10^{-5}$.

### 2. Phần 2: Phân Rã Ma Trận và Trực Quan Hóa

- **Source:** Nằm trong thư mục `part2/`. Bao gồm thuật toán Phân rã QR (Gram-Schmidt) và Chéo hóa ma trận (sử dụng `numpy.linalg.eigvals` cho đa thức bậc $\ge 5$ theo định lý Abel).
- **Chạy thử test case:** Có thể chạy trực tiếp các file python để xem output của bộ 5 Test Cases kiểm thử tự động:
  ```bash
  python part2/decomposition.py
  python part2/diagonalization.py
  ```

### 3. Phần 3: Giải hệ phương trình và Phân tích hiệu năng

- **Source:** Nằm trong thư mục `part3/`. Các hàm đo lường thời gian (dùng `time.perf_counter()`), sinh ma trận Hilbert, ma trận SPD và logic vòng lặp Gauss-Seidel.
- **Notebook:** File `part3/analysis.ipynb`. Notebook này chứa toàn bộ quy trình sinh dữ liệu từ kích thước $N=50$ đến $1000$, vẽ đồ thị log-log so sánh thời gian thực thi $\mathcal{O}(n^3)$ và đo lường sự khuếch đại sai số thông qua Số điều kiện (Condition Number).
