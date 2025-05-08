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

![BFS Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/bfs.gif)

DFS: Tìm kiếm theo chiều sâu.

![DFS Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/dfs.gif)

IDDFS: Tìm kiếm theo chiều sâu lặp.

![IDDFS Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/iddfs.gif)

UCS: Tìm kiếm chi phí đồng nhất.

![UCS Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/ucs.gif)

Backtracking: Tìm kiếm quay lui.

![Backtracking Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/backtrack.gif)

CSP Backtracking: Tìm kiếm quay lui với ràng buộc.

![CSP Backtracking Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/csp_bt.gif)

Sensorless BFS: Tìm kiếm không cảm biến.

![Sensorless BFS Backtracking Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/sensorless.gif)

Sensor BFS: Tìm kiếm với cảm biến.

![Sensor BFS Backtracking Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/sensor_bfs.gif)

1.3 Hình ảnh so sánh hiệu suất

Chương trình tạo ra hai biểu đồ so sánh hiệu suất (thời gian giải và số bước) của các thuật toán, được lưu dưới dạng hình ảnh:

![Comparison Time Image](https://raw.githubusercontent.com/tipoffkill/Python/main/comparison_time.png)

So sánh thời gian giải của các thuật toán.

![Comparison Steps Image](https://raw.githubusercontent.com/tipoffkill/Python/main/comparison_steps.png)

So sánh số bước của các thuật toán.

1.4 Nhận xét về hiệu suất

BFS và UCS thường tìm được đường đi tối ưu nhưng tốn nhiều thời gian hơn khi không gian trạng thái lớn.

DFS và IDDFS nhanh hơn nhưng không đảm bảo đường đi ngắn nhất, đôi khi bị kẹt ở độ sâu lớn.

Backtracking và CSP Backtracking có hiệu suất tương tự nhau, nhưng dễ bị giới hạn độ sâu.

Sensorless BFS và Sensor BFS tốn nhiều tài nguyên hơn do phải xử lý tập hợp trạng thái (belief states).

2. Các thuật toán tìm kiếm có thông tin

2.1 Trình bày các thành phần chính của bài toán tìm kiếm và giải pháp

Bài toán tìm kiếm: Tương tự như trên, nhưng các thuật toán sử dụng hàm heuristic (Manhattan Distance) để định hướng tìm kiếm.

Hàm heuristic: Tổng khoảng cách Manhattan của mỗi ô số so với vị trí đích của nó.

Giải pháp:

Sử dụng các thuật toán có thông tin (Greedy, A*, IDA*, Hill Climbing, Steepest Ascent Hill Climbing, Stochastic Hill Climbing, Simulated Annealing, Beam Search, Genetic Algorithm, Q-Learning) để tìm đường đi hiệu quả hơn.

Đường đi vẫn là chuỗi các hành động (U, D, L, R).
