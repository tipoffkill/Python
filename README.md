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

2.2 Hình ảnh GIF của từng thuật toán khi áp dụng lên trò chơi

Các thuật toán có thông tin được triển khai:

Greedy: Tìm kiếm tham lam dựa trên heuristic.

![Greedy Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/.gif)

A*: Tìm kiếm tốt nhất dựa trên chi phí và heuristic.

![A* Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/a_.gif)

IDA*: Tìm kiếm A* lặp.

![IDA* Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/ida_.gif)

Hill Climbing: Leo đồi cơ bản.

![Hill Climbing Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/hill.gif)

Steepest Ascent Hill Climbing: Leo đồi chọn bước tốt nhất.

![Steepest Ascent Hill Climbing Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/steepest.gif)

Stochastic Hill Climbing: Leo đồi ngẫu nhiên.

![ Ascent Hill Climbing Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/stochastic.gif)

Simulated Annealing: Giả lập ủ nhiệt.

![Simulated Annealing Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/sim.anneal.gif)

Beam Search: Tìm kiếm chùm.

![Beam Search Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/beam.gif)

Genetic Algorithm: Thuật toán di truyền.

![Genetic Algorithm Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/genetic.gif)

Q-Learning: Học tăng cường.

![Q-Learning Animation](https://raw.githubusercontent.com/tipoffkill/Python/main/gifs/q-learning.gif)

2.3 Hình ảnh so sánh hiệu suất

Tương tự như trên, các biểu đồ comparison_time.png và comparison_steps.png bao gồm cả các thuật toán có thông tin.

![Comparison Time Image](https://raw.githubusercontent.com/tipoffkill/Python/main/comparison_time.png)

So sánh thời gian giải của các thuật toán.

![Comparison Steps Image](https://raw.githubusercontent.com/tipoffkill/Python/main/comparison_steps.png)

So sánh số bước của các thuật toán.

2.4 Nhận xét về hiệu suất

A* và IDA* thường cho đường đi tối ưu và hiệu quả hơn các thuật toán không có thông tin.

Greedy nhanh nhưng không đảm bảo đường đi tối ưu.

Hill Climbing và các biến thể dễ bị kẹt ở cực trị cục bộ.

Simulated Annealing và Genetic Algorithm có thể tìm giải pháp tốt nhưng tốn thời gian và phụ thuộc vào tham số.

Q-Learning cần thời gian huấn luyện lâu nhưng có thể tái sử dụng kiến thức.

Công nghệ sử dụng

Ngôn ngữ: Python

Thư viện:

Pygame: Tạo giao diện và hoạt hình.

Matplotlib: Vẽ biểu đồ so sánh hiệu suất.

Môi trường: Python 3.8 trở lên

Hướng dẫn cài đặt

Clone repository:

git clone https://github.com/your-username/8-puzzle-solver.git


Cài đặt các thư viện cần thiết:

pip install pygame matplotlib

Chạy giao diện chính:

python solver.py

Chạy kiểm tra hiệu suất và tạo biểu đồ:

python test.py

Hướng dẫn sử dụng

Mở file solver.py để chạy giao diện chính.

Nhấn vào các nút thuật toán (BFS, A*, Hill, v.v.) để xem quá trình giải và hoạt hình.

Kiểm tra file test.py để chạy kiểm tra hiệu suất và tạo biểu đồ so sánh.

Tác giả

[Bùi Thành Tâm] - MSSV: [23110310]
