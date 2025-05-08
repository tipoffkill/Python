# Python
8-Puzzle Solver

Mục tiêu

Đồ án này nhằm xây dựng một chương trình giải bài toán 8-Puzzle (trò chơi ghép hình 3x3) sử dụng các thuật toán tìm kiếm không có thông tin và có thông tin. Chương trình cung cấp giao diện trực quan để hiển thị quá trình giải, tạo hoạt hình cho từng bước di chuyển, và so sánh hiệu suất giữa các thuật toán.

Nội dung

1. Các thuật toán tìm kiếm không có thông tin

1.1 Trình bày các thành phần chính của bài toán tìm kiếm và giải pháp

Bài toán tìm kiếm:

Không gian trạng thái: Mỗi trạng thái là một cách sắp xếp các ô số (0-8) trên bàn cờ 3x3, trong đó 0 là ô trống.

Trạng thái ban đầu: Một cấu hình ngẫu nhiên của các ô số (ví dụ: (2, 6, 5, 1, 3, 8, 4, 7, 0)).

Trạng thái đích: Cấu hình mục tiêu (1, 2, 3, 4, 5, 6, 7, 8, 0).

Hành động: Di chuyển ô trống lên (U), xuống (D), trái (L), hoặc phải (R) để hoán đổi với ô bên cạnh.

Chi phí: Mỗi bước di chuyển có chi phí là 1.

Giải pháp:

Sử dụng các thuật toán không có thông tin (BFS, DFS, IDDFS, UCS, Backtracking, CSP Backtracking, Sensorless BFS, Sensor BFS) để tìm đường đi từ trạng thái ban đầu đến trạng thái đích.

Đường đi là một chuỗi các hành động (U, D, L, R) dẫn đến trạng thái đích.

1.2 Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi

BFS: Tìm kiếm theo chiều rộng.
