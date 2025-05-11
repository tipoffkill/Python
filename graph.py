import sys
import signal
import platform
import traceback
import random
import time
import json
import threading
from functools import wraps
from contextlib import contextmanager

# --- Cấu hình ---
try:
    import solver
    print("Đã nhập thành công file 'solver.py'")
except ImportError:
    print("LỖI: Không thể nhập 'solver.py'.")
    print("Hãy đảm bảo file 'solver.py' tồn tại trong cùng thư mục.")
    sys.exit(1)
except Exception as e:
    print(f"LỖI khi nhập 'solver.py': {e}")
    sys.exit(1)

INITIAL_STATE = solver.d
GOAL_STATE = solver.g
ALGORITHMS_TO_TEST = [
    "BFS", "DFS", "IDDFS", "UCS", "Greedy", "A*", "IDA*",
    "Hill", "Steepest", "Stochastic", "Beam", "Sim. Anneal",
    "Genetic", "Backtrack"
]
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3  # Maximum retries for failed algorithms
RESULT_FILE = "algorithm_results.json"

# --- Định nghĩa bản đồ ánh xạ thuật toán ---
test_algo_map = {
    "BFS": solver.bfs,
    "DFS": lambda start, goal: solver.dfs(start, goal, max_depth=35),
    "IDDFS": lambda start, goal: solver.iddfs(start, goal, max_limit=50),
    "UCS": solver.ucs,
    "Greedy": solver.greedy,
    "A*": solver.astar,
    "IDA*": solver.ida_star,
    "Hill": solver.hill_climbing,
    "Steepest": solver.steepest_ascent_hill_climbing,
    "Stochastic": lambda start, goal: solver.stochastic_hill_climbing(start, goal, max_iterations=5000),
    "Beam": lambda start, goal: solver.beam_search(start, goal, beam_width=10, max_steps=150),
    "Sim. Anneal": lambda start, goal: solver.simulated_annealing(start, goal, initial_temp=1000, cooling_rate=0.99, min_temp=1e-6),
    "Genetic": lambda start, goal: solver.genetic_algorithm(start, goal, population_size=100, max_generations=300, mutation_rate=0.2, path_length=60),
    "Backtrack": lambda start, goal: solver.backtracking(start, goal, max_depth=35),
}

# --- Cross-Platform Timeout ---
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    if platform.system() == "Windows":
        def run_with_timeout(func, *args, **kwargs):
            result = [None]
            exception = [None]
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    exception[0] = e
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            if thread.is_alive():
                raise TimeoutError(f"Quá thời gian {seconds} giây!")
            if exception[0]:
                raise exception[0]
            return result[0]
        yield run_with_timeout
    else:
        def signal_handler(signum, frame):
            raise TimeoutError(f"Quá thời gian {seconds} giây!")
        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield lambda func, *args, **kwargs: func(*args, **kwargs)
        finally:
            signal.alarm(0)

# --- Hàm lưu kết quả ---
def save_results(results, filename=RESULT_FILE):
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Đã lưu kết quả vào '{filename}'")
    except Exception as e:
        print(f"LỖI khi lưu kết quả: {e}")

# --- Hàm chạy một thuật toán ---
def run_single_test(algo_name, start_state, goal_state, attempt=1):
    print(f"--- Đang kiểm tra: {algo_name} (Lần thử {attempt}) ---")
    if algo_name not in test_algo_map:
        print(f"   LỖI: Thuật toán '{algo_name}' không có trong test_algo_map.")
        return {
            "algo": algo_name,
            "time": 0.0,
            "steps": -1,
            "result": "Lỗi Mapping",
            "error": True,
            "attempt": attempt
        }

    algo_func = test_algo_map[algo_name]
    result_path = "Chưa chạy"
    solve_time = 0.0
    steps = -1
    error_occurred = False
    error_message = ""

    start_wall_time = time.time()
    try:
        with timeout(TIMEOUT_SECONDS) as run:
            result_path, solve_time_reported = run(algo_func, start_state, goal_state)
        solve_time = time.time() - start_wall_time

        valid_moves = {'U', 'D', 'L', 'R'}
        is_valid_path = isinstance(result_path, str) and result_path and all(m in valid_moves for m in result_path)

        if is_valid_path:
            steps = len(result_path)
            print(f"   Thành công! Thời gian: {solve_time:.4f}s, Số bước: {steps}")
        else:
            error_occurred = True
            error_message = result_path
            print(f"   Thất bại/Không tìm thấy. Thời gian: {solve_time:.4f}s, Kết quả: {result_path}")

    except TimeoutError as e:
        solve_time = time.time() - start_wall_time
        error_occurred = True
        error_message = f"Timeout ({TIMEOUT_SECONDS}s)"
        print(f"   LỖI: {error_message}. Thời gian: {solve_time:.4f}s")
    except Exception as e:
        solve_time = time.time() - start_wall_time
        error_occurred = True
        error_message = f"Lỗi Runtime: {type(e).__name__}: {str(e)}"
        print(f"   LỖI: {error_message} khi chạy {algo_name}.")
        print(traceback.format_exc())

    return {
        "algo": algo_name,
        "time": solve_time,
        "steps": steps,
        "result": result_path if not error_occurred else error_message,
        "error": error_occurred,
        "attempt": attempt
    }

# --- Hàm vẽ biểu đồ ---
def plot_results(results):
    try:
        import matplotlib.pyplot as plt
        print("\n--- Đang vẽ biểu đồ ---")
    except ImportError:
        print("\nLỖI: Không tìm thấy thư viện 'matplotlib'.")
        print("Hãy cài đặt bằng lệnh: pip install matplotlib")
        return

    successful_results = [r for r in results if not r["error"]]
    if not successful_results:
        print("Không có thuật toán nào hoàn thành thành công. Không thể vẽ đồ thị.")
        return

    algos = [r["algo"] for r in successful_results]
    times = [r["time"] for r in successful_results]
    steps = [r["steps"] for r in successful_results]

    sorted_indices = sorted(range(len(times)), key=lambda k: times[k])
    algos = [algos[i] for i in sorted_indices]
    times = [times[i] for i in sorted_indices]
    steps = [steps[i] for i in sorted_indices]

    plt.style.use('default')

    # --- Biểu đồ thời gian ---
    plt.figure(figsize=(12, 7))
    bars_time = plt.bar(algos, times, color='skyblue')
    plt.ylabel("Thời gian giải (giây)")
    plt.xlabel("Thuật toán")
    plt.title(f"So sánh thời gian giải 8-Puzzle\nTrạng thái bắt đầu: {solver.format_state_tuple(INITIAL_STATE)}")
    plt.xticks(rotation=45, ha='right')
    for bar in bars_time:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("comparison_time.png")
    print("Đã lưu biểu đồ thời gian vào 'comparison_time.png'")

    # --- Biểu đồ số bước ---
    plt.figure(figsize=(12, 7))
    bars_steps = plt.bar(algos, steps, color='lightcoral')
    plt.ylabel("Số bước (Độ dài đường đi)")
    plt.xlabel("Thuật toán")
    plt.title(f"So sánh số bước giải 8-Puzzle\nTrạng thái bắt đầu: {solver.format_state_tuple(INITIAL_STATE)}")
    plt.xticks(rotation=45, ha='right')
    for bar in bars_steps:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{int(yval)}', va='bottom', ha='center', fontsize=9)
    plt.tight_layout()
    plt.savefig("comparison_steps.png")
    print("Đã lưu biểu đồ số bước vào 'comparison_steps.png'")

    plt.show()

# --- Hàm chính ---
if __name__ == "__main__":
    print("="*50)
    print(" BẮT ĐẦU KIỂM TRA HIỆU SUẤT THUẬT TOÁN 8-PUZZLE")
    print("="*50)
    print(f"Trạng thái bắt đầu: {solver.format_state_tuple(INITIAL_STATE)}")
    print(f"Trạng thái đích:    {solver.format_state_tuple(GOAL_STATE)}")
    print(f"Timeout mỗi thuật toán: {TIMEOUT_SECONDS} giây")
    print(f"Số lần thử lại tối đa: {MAX_RETRIES}")

    if not solver.is_solvable(INITIAL_STATE, GOAL_STATE):
        print("\nLỖI: Trạng thái bắt đầu không thể giải được đến trạng thái đích!")
        sys.exit(1)
    else:
        print("Trạng thái bắt đầu có thể giải được.")

    all_results = []
    algorithms_to_process = ALGORITHMS_TO_TEST.copy()
    retry_counts = {algo: 0 for algo in ALGORITHMS_TO_TEST}

    while algorithms_to_process:
        current_round_results = []
        failed_algorithms = []

        for algo_name in algorithms_to_process:
            if algo_name not in test_algo_map:
                print(f"\n--- Bỏ qua: {algo_name} (Không tìm thấy trong test_algo_map) ---")
                continue

            attempt = retry_counts[algo_name] + 1
            result = run_single_test(algo_name, INITIAL_STATE, GOAL_STATE, attempt)
            current_round_results.append(result)
            all_results.append(result)
            save_results(all_results)

            if result["error"] and retry_counts[algo_name] < MAX_RETRIES:
                retry_counts[algo_name] += 1
                failed_algorithms.append(algo_name)
                print(f"   Thuật toán {algo_name} thất bại, sẽ thử lại (lần {retry_counts[algo_name]}/{MAX_RETRIES})")
            elif result["error"]:
                print(f"   Thuật toán {algo_name} thất bại sau {MAX_RETRIES} lần thử. Bỏ qua.")

        algorithms_to_process = failed_algorithms
        if algorithms_to_process:
            print(f"\n--- Còn {len(algorithms_to_process)} thuật toán cần thử lại: {algorithms_to_process} ---")

    # In bảng tổng kết
    print("\n" + "="*60)
    print(" KẾT QUẢ KIỂM TRA TỔNG HỢP")
    print("="*60)
    print(f"{'Thuật toán':<15} | {'Thời gian (s)':<15} | {'Số bước':<10} | {'Kết quả/Lỗi':<30}")
    print("-"*60)
    for r in all_results:
        time_str = f"{r['time']:.4f}" if not r['error'] else "N/A"
        steps_str = str(r['steps']) if r['steps'] != -1 else "N/A"
        result_str = r['result']
        if not r['error'] and len(result_str) > 25:
            result_str = result_str[:25] + "..."
        print(f"{r['algo']:<15} | {time_str:<15} | {steps_str:<10} | {result_str:<30}")
    print("="*60)

    # Vẽ biểu đồ
    plot_results(all_results)

    print("\nHoàn thành kiểm tra và vẽ biểu đồ.")