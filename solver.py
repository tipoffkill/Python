import pygame
import sys
import time
import heapq
import random
from collections import deque
import math
import imageio
import numpy as np
import os


pygame.init()
w, h = 700, 700
s = pygame.display.set_mode((w, h))
pygame.display.set_caption("8-Puzzle Solver")

# --- Colors ---
c1 = (245, 245, 220)  # Light background (Beige)
c2 = (40, 40, 40)     # Dark gray 
c3 = (255, 180, 180)  # Light pink 
c4 = (70, 130, 180)   # Steel blue 
c5 = (255, 140, 105)  # Coral 
c6 = (255, 100, 100)  # Light red 
c_pink = (255, 200, 200)  # Pink 
c_bg = (220, 220, 220)  # Light gray 
c_red_border = (255, 0, 0) # Red border 

f = pygame.font.Font(None, 40) 
f2 = pygame.font.Font(None, 26)
f_info = pygame.font.Font(None, 22) 
f_path = pygame.font.Font(None, 20) 

# Trạng thái khởi đầu bạn cung cấp
d = (2, 6, 5, 1, 3, 8, 4, 7, 0)
g = (1, 2, 3, 4, 5, 6, 7, 8, 0)

current_algo = ""       
current_result = ""     
solve_time_display = 0.0 
display_time_display = 0.0 
animation_active = False 
buttons = {}            
tiles = []              

def drawText(surface, text, color, rect, font, aa=True, bkg=None):
    rect = pygame.Rect(rect)
    y = rect.top
    lineSpacing = -2

    fontHeight = font.size("Tg")[1]

    while text:
        i = 1

        if y + fontHeight > rect.bottom:
            break

        while font.size(text[:i])[0] < rect.width and i < len(text):
            i += 1

        if i < len(text):
            space_pos = text.rfind(" ", 0, i) + 1
            if space_pos > 0:
                i = space_pos
          
            else:
                 while font.size(text[:i])[0] > rect.width and i > 1:
                     i -= 1
                 if i == 1 and font.size(text[:i])[0] > rect.width:
                     print(f"Warning: Cannot fit character '{text[0]}' in width {rect.width}")
                     i = 0
        if i == 0:
             break 

        if bkg:
            image = font.render(text[:i], 1, color, bkg)
            image.set_colorkey(bkg)
        else:
            image = font.render(text[:i], aa, color)

        surface.blit(image, (rect.left, y))
        y += fontHeight + lineSpacing

        text = text[i:]

    return y 

class Tile:
    def __init__(self, value, x, y):
        self.value = value
        self.rect = pygame.Rect(x, y, 110, 110)
        self.target_x = x
        self.target_y = y
        self.speed = 15 # Animation speed

    def move(self):
        dx = self.target_x - self.rect.x
        dy = self.target_y - self.rect.y
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > self.speed:
            # Move proportionally
            self.rect.x += self.speed * dx / dist
            self.rect.y += self.speed * dy / dist
        else:
            self.rect.x = self.target_x
            self.rect.y = self.target_y

    def is_moving(self):
        return self.rect.x != self.target_x or self.rect.y != self.target_y

    def draw(self, surface, highlight=False):
        if self.value == 0: color = c3
        elif highlight: color = c6
        else: color = c_pink
        pygame.draw.rect(surface, color, self.rect, border_radius=10)
        pygame.draw.rect(surface, c2, self.rect, 2, border_radius=10)
        if self.value:
            text = f.render(str(self.value), True, c2)
            text_rect = text.get_rect(center=self.rect.center)
            surface.blit(text, text_rect)

def draw_grid(surface):
    for i in range(3):
        for j in range(3):
            bg_rect = pygame.Rect(j * 120 + 50 - 2, i * 120 + 50 - 2, 110 + 4, 110 + 4)
            pygame.draw.rect(surface, c_bg, bg_rect, border_radius=12)

def draw(current_tiles):
    """Main drawing function."""
    s.fill(c1)
    draw_grid(s)

    moving_tile_value = -1
    is_animating_now = False
    if animation_active:
        for tile in current_tiles:
            if tile.value != 0 and tile.is_moving():
                moving_tile_value = tile.value
                is_animating_now = True
                break

    for tile in current_tiles:
        highlight = (is_animating_now and tile.value == moving_tile_value)
        tile.draw(s, highlight=highlight)

    draw_buttons()
    draw_status(solve_time_display, display_time_display)
    pygame.display.flip()

def draw_buttons():
    global buttons
    buttons.clear()
    btn_w, btn_h = 120, 38 
    start_x, start_y = 450, 30 
    spacing = 45
    cols = 2
    col_width = btn_w + 10

    button_list = [
        ("BFS", c4), ("DFS", c5), ("IDDFS", c4), ("UCS", c5),
        ("Greedy", c4), ("A*", c5), ("IDA*", c4),
        ("Hill", c5), ("Steepest", c4), ("Stochastic", c5),
        ("Beam", c4), ("Sim. Anneal", c5), ("Genetic", c4),
        ("Backtrack", c5), ("CSP BT", c4), ("Q-Learning", c5),
        ("Sensorless", c4), ("Sensor BFS", c5),
    ]

    max_rows = 10 
    for idx, (text, color) in enumerate(button_list):
        col = idx // max_rows
        row = idx % max_rows
        x = start_x + col * col_width
        y = start_y + row * spacing
        rect = pygame.Rect(x, y, btn_w, btn_h)

        border_color = c2
        border_width = 1
        normalized_current_algo = "Sim. Anneal" if current_algo == "Simulated Annealing" else current_algo
        if text == normalized_current_algo:
             border_color = c_red_border
             border_width = 3

        pygame.draw.rect(s, color, rect, border_radius=5)
        pygame.draw.rect(s, border_color, rect, border_width, border_radius=5)
        t = f2.render(text, True, c1)
        text_rect = t.get_rect(center=rect.center)
        s.blit(t, text_rect)
        buttons[text] = rect

def format_state_tuple(state_tuple):
    return str(state_tuple).replace(' ', '')

def draw_status(solve_time, display_time):
    info_box_rect = pygame.Rect(20, h - 250, 400, 230)
    pygame.draw.rect(s, c_bg, info_box_rect, border_radius=10)
    pygame.draw.rect(s, c2, info_box_rect, 1, border_radius=10)

    start_str = format_state_tuple(d)
    goal_str = format_state_tuple(g)

    try:
        line_height = f_info.get_linesize()
    except AttributeError:
        line_height = 22
    try:
        path_line_height = f_path.get_linesize()
    except AttributeError:
        path_line_height = 20

    padding = 10
    current_y = info_box_rect.top + padding

    text_surf = f_info.render(f"Start: {start_str}", True, c2)
    s.blit(text_surf, (info_box_rect.left + padding, current_y))
    current_y += line_height

    text_surf = f_info.render(f"Goal:  {goal_str}", True, c2)
    s.blit(text_surf, (info_box_rect.left + padding, current_y))
    current_y += line_height // 2

    sep_y = current_y + 6
    pygame.draw.line(s, c2, (info_box_rect.left + padding, sep_y), (info_box_rect.right - padding, sep_y), 1)
    current_y += line_height // 2 + 6

    if not current_algo:
        algo_str = "Algorithm: -"
        result_prefix = "Result: "
        result_str = "-" 
        steps_str = "Steps: -"
        solve_str = "Solve Time: -"

        text_surf = f_info.render(algo_str, True, c2)
        s.blit(text_surf, (info_box_rect.left + padding, current_y))
        current_y += line_height

        prefix_surf = f_info.render(result_prefix, True, c2)
        s.blit(prefix_surf, (info_box_rect.left + padding, current_y))
        result_text_x = info_box_rect.left + padding + prefix_surf.get_width()
        result_surf = f_info.render(result_str, True, c2) 
        s.blit(result_surf, (result_text_x, current_y))
        current_y += line_height 

        if current_y < info_box_rect.bottom - padding - line_height:
            text_surf = f_info.render(steps_str, True, c2)
            s.blit(text_surf, (info_box_rect.left + padding, current_y))
            current_y += line_height
        
        if current_y < info_box_rect.bottom - padding:
            text_surf = f_info.render(solve_str, True, c2)
            s.blit(text_surf, (info_box_rect.left + padding, current_y))

    else:
        text_surf = f_info.render(f"Algorithm: {current_algo}", True, c2)
        s.blit(text_surf, (info_box_rect.left + padding, current_y))
        current_y += line_height 

        result_prefix = "Result: "
        prefix_surf = f_info.render(result_prefix, True, c2) 
        s.blit(prefix_surf, (info_box_rect.left + padding, current_y))

        result_text_x = info_box_rect.left + padding + prefix_surf.get_width()
        result_width = info_box_rect.width - (result_text_x - info_box_rect.left) - padding
        min_space_below = line_height * 2 + padding * 1.5 
        result_height = info_box_rect.bottom - current_y - min_space_below
        result_height = max(path_line_height, result_height)
        result_rect = pygame.Rect(result_text_x, current_y, result_width, result_height)
        font_to_use = f_path 

        result_display_str = current_result if current_result else "N/A"
        y_after_result = drawText(s, result_display_str, c2, result_rect, font_to_use)
        current_y = y_after_result + line_height // 3
 
        valid_moves = {'U', 'D', 'L', 'R'}
        is_path = isinstance(current_result, str) and current_result and all(m in valid_moves for m in current_result)
        if is_path:
            num_steps = len(current_result)
            steps_str = f"Steps: {num_steps}"
        else:
            steps_str = "Steps: N/A" 

        if current_y < info_box_rect.bottom - padding - line_height:
            text_surf = f_info.render(steps_str, True, c2)
            s.blit(text_surf, (info_box_rect.left + padding, current_y))
            current_y += line_height
       
        solve_str = f"Solve Time: {solve_time:.4f} s"
        if current_y < info_box_rect.bottom - padding:
            text_surf = f_info.render(solve_str, True, c2)
            s.blit(text_surf, (info_box_rect.left + padding, current_y))
        
def get_neighbors(state):
    neighbors = []
    try: zero_idx = state.index(0)
    except ValueError: return []
    r, c = zero_idx // 3, zero_idx % 3
    moves = [(-1, 0, 'U'), (1, 0, 'D'), (0, -1, 'L'), (0, 1, 'R')]
    for dr, dc, move_char in moves:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            new_state_list = list(state)
            new_idx = nr * 3 + nc
            new_state_list[zero_idx], new_state_list[new_idx] = new_state_list[new_idx], new_state_list[zero_idx]
            neighbors.append((move_char, tuple(new_state_list)))
    return neighbors

def heuristic(state, goal=g):
    distance = 0
    for i, val in enumerate(state):
        if val != 0:
            try: goal_idx = goal.index(val)
            except ValueError: return float('inf')
            current_row, current_col = i // 3, i % 3
            goal_row, goal_col = goal_idx // 3, goal_idx % 3
            distance += abs(current_row - goal_row) + abs(current_col - goal_col)
    return distance

def apply_moves(state, moves):
    current = list(state)
    for move in moves:
        try: zero_idx = current.index(0)
        except ValueError: return None # Error
        r, c = zero_idx // 3, zero_idx % 3
        new_idx = -1
        if move == 'U' and r > 0: new_idx = zero_idx - 3
        elif move == 'D' and r < 2: new_idx = zero_idx + 3
        elif move == 'L' and c > 0: new_idx = zero_idx - 1
        elif move == 'R' and c < 2: new_idx = zero_idx + 1
        if new_idx != -1:
            current[zero_idx], current[new_idx] = current[new_idx], current[zero_idx]
    return tuple(current)

def is_solvable(state, goal=g):
    state_list = [x for x in state if x != 0]
    goal_list = [x for x in goal if x != 0]
    def count_inversions(arr):
        count = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]: count += 1
        return count
    return (count_inversions(state_list) % 2) == (count_inversions(goal_list) % 2)

def generate_solvable_states(goal, num_states=3):
    states = set()
    attempts = 0
    max_attempts = num_states * 100
    while len(states) < num_states and attempts < max_attempts:
        state_list = list(goal)
        random.shuffle(state_list)
        state = tuple(state_list)
        if state != goal and is_solvable(state, goal): states.add(state)
        attempts += 1
    if len(states) < num_states: print(f"Warning: Generated only {len(states)} states.")
    return states

# --- Algorithm Implementations ---
# BFS
def bfs(start_state, goal_state):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    queue = deque([("", start_state)])
    visited = {start_state}
    while queue:
        path, current_state = queue.popleft()
        if current_state == goal_state:
            return path, time.time() - start_time
        for move, next_state in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                queue.append((path + move, next_state))
    return "No Solution", time.time() - start_time

# DFS
def dfs(start_state, goal_state, max_depth=35):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    stack = [("", start_state, 0)]
    visited = {}
    while stack:
        path, current_state, depth = stack.pop()
        if current_state == goal_state: return path, time.time() - start_time
        if current_state in visited and visited[current_state] <= depth: continue
        visited[current_state] = depth
        if depth < max_depth:
            for move, next_state in reversed(get_neighbors(current_state)):
                 if next_state not in visited or visited[next_state] > depth + 1:
                    stack.append((path + move, next_state, depth + 1))
    return f"No Solution (Depth {max_depth})", time.time() - start_time

# UCS
def ucs(start_state, goal_state):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    priority_queue = [(0, "", start_state)]
    visited = {start_state: 0}
    while priority_queue:
        cost, path, current_state = heapq.heappop(priority_queue)
        if current_state == goal_state: return path, time.time() - start_time
        if cost > visited[current_state]: continue
        for move, next_state in get_neighbors(current_state):
            new_cost = cost + 1
            if next_state not in visited or new_cost < visited[next_state]:
                visited[next_state] = new_cost
                heapq.heappush(priority_queue, (new_cost, path + move, next_state))
    return "No Solution", time.time() - start_time

# IDDFS
def iddfs(start_state, goal_state, max_limit=50):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    def dls(path, current_state, depth_limit, visited_in_path):
        if current_state == goal_state: return path
        if len(path) >= depth_limit: return None
        visited_in_path.add(current_state)
        for move, next_state in get_neighbors(current_state):
            if next_state not in visited_in_path:
                result = dls(path + move, next_state, depth_limit, visited_in_path)
                if result is not None:
                    visited_in_path.remove(current_state)
                    return result
        visited_in_path.remove(current_state) 
        return None
    for depth in range(max_limit + 1):
        result = dls("", start_state, depth, set())
        if result is not None: return result, time.time() - start_time
    return f"No Solution (Limit {max_limit})", time.time() - start_time

# Greedy
def greedy(start_state, goal_state):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    priority_queue = [(heuristic(start_state, goal_state), "", start_state)]
    visited = {start_state}
    exp_count = 0
    max_exp = 10000
    while priority_queue and exp_count < max_exp:
        _, path, current_state = heapq.heappop(priority_queue)
        exp_count += 1
        if current_state == goal_state: return path, time.time() - start_time
        for move, next_state in get_neighbors(current_state):
            if next_state not in visited:
                visited.add(next_state)
                h_cost = heuristic(next_state, goal_state)
                heapq.heappush(priority_queue, (h_cost, path + move, next_state))
    return f"Timeout/Stuck (Greedy, {exp_count} exp)", time.time() - start_time

# A*
def astar(start_state, goal_state):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    h_start = heuristic(start_state, goal_state)
    priority_queue = [(h_start, 0, "", start_state)] # f, g, path, state
    visited = {start_state: 0}
    while priority_queue:
        f_cost_est, g_cost, path, current_state = heapq.heappop(priority_queue)
        if current_state == goal_state: return path, time.time() - start_time
        if g_cost > visited[current_state]: continue
        for move, next_state in get_neighbors(current_state):
            new_g_cost = g_cost + 1
            if next_state not in visited or new_g_cost < visited[next_state]:
                visited[next_state] = new_g_cost
                h_cost = heuristic(next_state, goal_state)
                f_cost = new_g_cost + h_cost
                heapq.heappush(priority_queue, (f_cost, new_g_cost, path + move, next_state))
    return "No Solution", time.time() - start_time

# IDA*
def ida_star(start_state, goal_state):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    def search(path, g_cost, bound, visited_in_path):
        current_state = apply_moves(start_state, path)
        if current_state is None: return float('inf'), None
        h_cost = heuristic(current_state, goal_state)
        f_cost = g_cost + h_cost
        if f_cost > bound: return f_cost, None
        if current_state == goal_state: return f_cost, path
        min_bound = float('inf')
        visited_in_path.add(current_state)
        for move, next_state in get_neighbors(current_state):
            if next_state not in visited_in_path:
                threshold, result = search(path + move, g_cost + 1, bound, visited_in_path)
                if result is not None:
                    visited_in_path.remove(current_state)
                    return threshold, result
                min_bound = min(min_bound, threshold)
        visited_in_path.remove(current_state)
        return min_bound, None
    bound = heuristic(start_state, goal_state)
    iter_count = 0
    max_iters = 100
    while iter_count < max_iters:
        threshold, result = search("", 0, bound, set())
        if result is not None: return result, time.time() - start_time
        if threshold == float('inf'): return "No Solution (IDA*)", time.time() - start_time
        bound = threshold
        iter_count += 1
    return f"Timeout (IDA*, {max_iters} iters)", time.time() - start_time

# --- Local Search & Metaheuristics 
def ke(x):
    n = []
    z = x.index(0)
    r, c = z // 3, z % 3
    for dr, dc, m in [(1, 0, 'D'), (-1, 0, 'U'), (0, 1, 'R'), (0, -1, 'L')]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            y = list(x)
            nz = nr * 3 + nc
            y[z], y[nz] = y[nz], y[z]
            n.append((m, tuple(y)))
    return n

def hill_climbing(d_start, g_goal): 
    start = time.time()
    current = d_start
    path = ""
    visited = {current} 

    while True:
        if current == g_goal: return path, time.time() - start

        neighbors = ke(current)
        random.shuffle(neighbors)

        found_better = False
        best_neighbor = None
        best_move = None
        current_h = heuristic(current, g_goal) 

        # Find the first better neighbor
        for move, next_state in neighbors:
            if next_state not in visited:
                 if heuristic(next_state, g_goal) < current_h:
                    best_neighbor = next_state
                    best_move = move
                    found_better = True
                    break # Take first better one

        if found_better:
            current = best_neighbor
            path += best_move
            visited.add(current)
        else:
            # No better unvisited neighbor found
            return f"Stuck (h={current_h}, {len(path)} moves)", time.time() - start

def steepest_ascent_hill_climbing(d_start, g_goal):
    start = time.time()
    current = d_start
    path = ""
    visited = {current}

    while True:
        if current == g_goal: return path, time.time() - start

        neighbors = ke(current)
        best_neighbor = None
        best_move = None
        best_h = heuristic(current, g_goal) # Current best heuristic

        # Find the absolute best neighbor
        for move, next_state in neighbors:
            if next_state not in visited:
                h = heuristic(next_state, g_goal)
                if h < best_h:
                    best_h = h
                    best_neighbor = next_state
                    best_move = move

        if best_neighbor is not None: # Found an improving move
            current = best_neighbor
            path += best_move
            visited.add(current)
        else:
            # No improving unvisited neighbor
            final_h = heuristic(current, g_goal)
            return f"Stuck (h={final_h}, {len(path)} moves)", time.time() - start

def stochastic_hill_climbing(d_start, g_goal, max_iterations=1000, accept_worse_prob=0.0): # Set worse prob to 0 for pure stochastic HC
    start = time.time()
    current = d_start
    path = ""
    iteration = 0
    best_state_found = current
    best_h_found = heuristic(current, g_goal)

    while iteration < max_iterations:
        if current == g_goal: return path, time.time() - start

        # Track best
        current_h = heuristic(current, g_goal)
        if current_h < best_h_found:
            best_h_found = current_h
            best_state_found = current

        neighbors = ke(current)
        if not neighbors: break 
        
        uphill_neighbors = []
        for move, next_state in neighbors:
            if heuristic(next_state, g_goal) < current_h:
                uphill_neighbors.append((move, next_state))

        if uphill_neighbors:
            move, next_state = random.choice(uphill_neighbors)
            current = next_state
            path += move
        elif accept_worse_prob > 0 and neighbors: 
             move, next_state = random.choice(neighbors)
             if random.random() < accept_worse_prob:
                 current = next_state
                 path += move

        iteration += 1

    final_h = heuristic(current, g_goal)
    return f"Timeout ({max_iterations} iter, Best h={best_h_found})", time.time() - start


def simulated_annealing(d_start, g_goal, initial_temp=1000, cooling_rate=0.95, min_temp=1e-5):
    start = time.time()
    current = d_start
    path = ""
    temp = initial_temp
    best_state = current
    best_h = heuristic(current, g_goal)

    step = 0
    max_steps = 15000

    while temp > min_temp and step < max_steps:
        step += 1
        if current == g_goal: return path, time.time() - start

        neighbors = ke(current)
        if not neighbors: break

        move, next_state = random.choice(neighbors)
        current_h = heuristic(current, g_goal)
        next_h = heuristic(next_state, g_goal)
        delta_h = next_h - current_h

        accepted = False
        if delta_h < 0:
            accepted = True
        elif random.random() < math.exp(-delta_h / temp):
            accepted = True

        if accepted:
            current = next_state
            path += move
            if next_h < best_h:
                best_h = next_h
                best_state = current

        temp *= cooling_rate

    final_h = heuristic(best_state, g_goal)
    return f"Timeout/Frozen (Best h={final_h})", time.time() - start


# Beam Search
def beam_search(start_state, goal_state, beam_width=5, max_steps=100):
    start_time = time.time()
    h_start = heuristic(start_state, goal_state)
    beam = [(h_start, "", start_state)]
    visited = {start_state}
    for step in range(max_steps):
        if not beam: return "No Solution (Empty Beam)", time.time() - start_time
        next_candidates = []
        found_goal = False
        solution_path = ""
        min_goal_cost = float('inf')
        for h_val, path, current in beam:
            if current == goal_state:
                if len(path) < min_goal_cost:
                     solution_path = path
                     min_goal_cost = len(path)
                found_goal = True
                continue
            for move, next_state in get_neighbors(current): # Use standard get_neighbors
                if next_state not in visited:
                    visited.add(next_state)
                    h_next = heuristic(next_state, goal_state)
                    heapq.heappush(next_candidates, (h_next, path + move, next_state))
        if found_goal: return solution_path, time.time() - start_time
        beam = []
        count = 0
        while next_candidates and count < beam_width:
            beam.append(heapq.heappop(next_candidates))
            count += 1
        if not beam: return "No Solution (Beam Dead End)", time.time() - start_time
    best_h_final = min(b[0] for b in beam) if beam else float('inf')
    return f"Timeout ({max_steps} steps, BW={beam_width}, Best h={best_h_final})", time.time() - start_time

# Genetic Algorithm
def genetic_algorithm(start_state, goal_state, population_size=50, max_generations=100, mutation_rate=0.1, elitism_size=5, path_length=40):
    start_time = time.time()
    moves = ['U', 'D', 'L', 'R']
    population = [[random.choice(moves) for _ in range(path_length)] for _ in range(population_size)]
    best_fitness_overall = float('inf')
    best_individual_overall = []
    for generation in range(max_generations):
        fitness_scores = []
        for individual in population:
            final_state = apply_moves(start_state, individual)
            if final_state is None: fitness = float('inf')
            elif final_state == goal_state: return "".join(individual), time.time() - start_time
            else: fitness = heuristic(final_state, goal_state)
            fitness_scores.append((fitness, individual))
            if fitness < best_fitness_overall:
                best_fitness_overall = fitness
                best_individual_overall = individual
        fitness_scores.sort(key=lambda x: x[0])
        new_population = [fitness_scores[i][1] for i in range(elitism_size)]
        while len(new_population) < population_size:
            k = min(5, len(fitness_scores))
            if k <=0: break
            parents1 = random.sample(fitness_scores, k)
            parent1 = min(parents1, key=lambda x: x[0])[1]
            parents2 = random.sample(fitness_scores, k)
            parent2 = min(parents2, key=lambda x: x[0])[1]
            if len(parent1) > 1 and len(parent2) > 1:
                crossover_point = random.randint(1, path_length - 1)
                child = parent1[:crossover_point] + parent2[crossover_point:]
            else: child = parent1[:] if parent1 else parent2[:]
            for i in range(path_length):
                if random.random() < mutation_rate and child: child[i] = random.choice(moves)
            new_population.append(child)
        population = new_population
    final_state_best = apply_moves(start_state, best_individual_overall)
    h_best = heuristic(final_state_best, goal_state) if final_state_best else float('inf')
    path_str = "".join(best_individual_overall)
    return f"Timeout ({max_generations} gens, Best h={h_best})", time.time() - start_time

# Backtracking
def backtracking(start_state, goal_state, max_depth=35):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    def backtrack_recursive(current_state, path, depth, visited_in_path):
        state_tuple = tuple(current_state)
        if state_tuple == goal_state: return path
        if depth >= max_depth: return None
        if state_tuple in visited_in_path: return None
        visited_in_path.add(state_tuple)
        for move, next_state_tuple in get_neighbors(state_tuple): # Use standard neighbors
            result = backtrack_recursive(list(next_state_tuple), path + move, depth + 1, visited_in_path) # Pass list
            if result is not None:
                visited_in_path.remove(state_tuple)
                return result
        visited_in_path.remove(state_tuple)
        return None
    result = backtrack_recursive(list(start_state), "", 0, set()) # Start with list
    if result is not None: return result, time.time() - start_time
    else: return f"No Solution (Depth {max_depth})", time.time() - start_time

# CSP Backtracking
def csp_backtracking(start_state, goal_state, max_depth=35):
    start_time = time.time()
    if start_state == goal_state: return "", time.time() - start_time
    def backtrack_csp_recursive(current_state, path, depth, visited_in_path):
        if current_state == goal_state: return path
        if depth >= max_depth: return None
        if current_state in visited_in_path: return None
        visited_in_path.add(current_state)
        for move, next_state in get_neighbors(current_state): # Use standard neighbors
            result = backtrack_csp_recursive(next_state, path + move, depth + 1, visited_in_path)
            if result is not None:
                visited_in_path.remove(current_state)
                return result
        visited_in_path.remove(current_state)
        return None
    result = backtrack_csp_recursive(start_state, "", 0, set()) # Start with tuple
    if result is not None: return result, time.time() - start_time
    else: return f"No Solution (Depth {max_depth})", time.time() - start_time

# Q-Learning
def q_learning(start_state, goal_state, episodes=20000, alpha=0.1, gamma=0.9, epsilon=0.2, max_steps_per_episode=150, max_path_length=100):
    start_time_total = time.time()
    Q = {}
    moves_q = ['U', 'D', 'L', 'R']
    print(f"Q-Learning: Training for {episodes} episodes...")
    start_time_train = time.time()
    for episode in range(episodes):
        state_list = list(start_state)
        for step in range(max_steps_per_episode):
            state_tuple = tuple(state_list)
            if state_tuple == goal_state: break
            valid_neighbors = get_neighbors(state_tuple) # Standard neighbors
            valid_actions = [m for m, ns in valid_neighbors]
            if not valid_actions: break
            action = None
            if random.random() < epsilon: action = random.choice(valid_actions)
            else:
                q_values = {a: Q.get((state_tuple, a), 0.0) for a in valid_actions}
                if q_values:
                    max_q = max(q_values.values())
                    best_actions = [a for a, q in q_values.items() if q == max_q]
                    action = random.choice(best_actions)
                else: action = random.choice(valid_actions)
            next_state_tuple = None
            for m, ns in valid_neighbors:
                if m == action: next_state_tuple = ns; break
            if next_state_tuple is None: continue
            reward = -1
            if next_state_tuple == goal_state: reward = 100
            current_q = Q.get((state_tuple, action), 0.0)
            next_valid_neighbors = get_neighbors(next_state_tuple)
            next_valid_actions = [m for m, ns in next_valid_neighbors]
            max_next_q = 0.0
            if next_valid_actions:
                 next_q_values = {a: Q.get((next_state_tuple, a), 0.0) for a in next_valid_actions}
                 if next_q_values: max_next_q = max(next_q_values.values())
            new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
            Q[(state_tuple, action)] = new_q
            state_list = list(next_state_tuple)
    train_time = time.time() - start_time_train
    print(f"Q-Learning: Training complete in {train_time:.2f}s. Q-table size: {len(Q)}. Generating path...")
    state = start_state
    path = ""
    visited_for_path = {state}
    for _ in range(max_path_length):
        if state == goal_state: return path, time.time() - start_time_total
        state_tuple = tuple(state)
        valid_neighbors = get_neighbors(state_tuple) # Standard neighbors
        valid_actions = [m for m, ns in valid_neighbors]
        if not valid_actions: return f"No Path (QL - Stuck, {len(path)} moves)", time.time() - start_time_total
        q_values = {a: Q.get((state_tuple, a), -float('inf')) for a in valid_actions}
        if not q_values or all(q == -float('inf') for q in q_values.values()):
             return f"No Path (QL - Unexplored/Stuck, {len(path)} moves)", time.time() - start_time_total
        else:
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            action = random.choice(best_actions)
        next_state = None
        for m, ns in valid_neighbors:
            if m == action: next_state = ns; break
        if next_state is None: return f"Path Error (QL - Action Invalid?, {len(path)} moves)", time.time() - start_time_total
        if next_state in visited_for_path: return f"No Path (QL - Loop Detected, {len(path)} moves)", time.time() - start_time_total
        visited_for_path.add(next_state)
        path += action
        state = next_state
    return f"Timeout Path (QL - {max_path_length} moves)", time.time() - start_time_total

# Sensorless BFS
def sensorless_bfs(initial_belief_set, goal_state):
    start_time = time.time()
    if not initial_belief_set: return "No Solution (Empty Belief)", time.time()
    if all(s == goal_state for s in initial_belief_set): return "", time.time()
    initial_belief_fset = frozenset(initial_belief_set)
    queue = deque([("", initial_belief_fset)])
    visited = {initial_belief_fset}
    moves_s = ['U', 'D', 'L', 'R']
    max_belief_size = 10
    step_count = 0
    max_steps = 500
    print(f"Sensorless BFS: Initial belief size = {len(initial_belief_set)}, Max size = {max_belief_size}")
    while queue and step_count < max_steps:
        step_count += 1
        path, current_belief_fset = queue.popleft()
        if all(s == goal_state for s in current_belief_fset):
            print(f"Sensorless BFS: Goal reached at step {step_count}")
            return path, time.time() - start_time
        for action in moves_s:
            next_belief_set = set()
            for state in current_belief_fset:
                action_applied = False
                for move, next_state in get_neighbors(state): # Standard neighbors
                    if move == action:
                        next_belief_set.add(next_state); action_applied = True; break
            if next_belief_set:
                pruned = False
                if len(next_belief_set) > max_belief_size:
                     next_belief_set = set(random.sample(list(next_belief_set), max_belief_size)); pruned = True
                next_belief_fset = frozenset(next_belief_set)
                if next_belief_fset not in visited:
                    visited.add(next_belief_fset)
                    queue.append((path + action, next_belief_fset))
    if step_count >= max_steps: return f"Timeout (Sensorless BFS, {max_steps} steps)", time.time() - start_time
    else: return "No Solution (Sensorless BFS)", time.time() - start_time

# Sensor BFS
def sensorless_bfs_sensor(initial_belief_set, goal_state):
    start_time = time.time()
    if not initial_belief_set: return "No Solution (Empty Belief)", time.time()
    if all(s == goal_state for s in initial_belief_set): return "", time.time()
    initial_belief_fset = frozenset(initial_belief_set)
    queue = deque([("", initial_belief_fset)])
    visited = {initial_belief_fset}
    moves_sb = ['U', 'D', 'L', 'R']
    max_belief_size = 10
    step_count = 0
    max_steps = 500
    print(f"Sensor BFS: Initial belief size = {len(initial_belief_set)}, Max size = {max_belief_size}")
    while queue and step_count < max_steps:
        step_count += 1
        path, current_belief_fset = queue.popleft()
        if all(s == goal_state for s in current_belief_fset):
             print(f"Sensor BFS: Goal reached at step {step_count}")
             return path, time.time() - start_time
        for action in moves_sb:
            possible_next_beliefs = {}
            for state in current_belief_fset:
                action_applied = False
                for move, next_state in get_neighbors(state): # Standard neighbors
                    if move == action:
                        try: blank_pos = next_state.index(0)
                        except ValueError: continue
                        if blank_pos not in possible_next_beliefs: possible_next_beliefs[blank_pos] = set()
                        possible_next_beliefs[blank_pos].add(next_state)
                        action_applied = True; break
            for sensed_blank_pos, next_belief_set in possible_next_beliefs.items():
                if next_belief_set:
                    pruned = False
                    if len(next_belief_set) > max_belief_size:
                         next_belief_set = set(random.sample(list(next_belief_set), max_belief_size)); pruned = True
                    next_belief_fset = frozenset(next_belief_set)
                    if next_belief_fset not in visited:
                        visited.add(next_belief_fset)
                        queue.append((path + action, next_belief_fset))
    if step_count >= max_steps: return f"Timeout (Sensor BFS, {max_steps} steps)", time.time() - start_time
    else: return "No Solution (Sensor BFS)", time.time() - start_time


# --- Execution and Animation ---

def run_algorithm(algo_func, algo_name, initial_state=d, goal_state=g):
    """Runs the selected algorithm and handles animation."""
    global current_algo, current_result, tiles, animation_active
    global solve_time_display, display_time_display 

    animation_active = False 

    # Reset status
    current_algo = algo_name
    current_result = "Running..."
    solve_time_display = 0.0
    display_time_display = 0.0 # Reset display time counter

    actual_start_state = initial_state
    tiles = [Tile(actual_start_state[i * 3 + j], j * 120 + 50, i * 120 + 50) for i in range(3) for j in range(3)]

    # Draw initial state and message
    draw(tiles)
    pygame.event.pump()

    # --- Run the selected algorithm ---
    solution_path = "Error"
    solve_time = 0.0
    pre_solve_time = time.time() # Time before calling the algorithm

    try:
        # Handle Sensorless/Q-Learning separately if needed (e.g., belief state gen)
        if algo_name in ["Sensorless", "Sensor BFS"]:
             print(f"Generating belief state for {algo_name}...")
             belief_states = generate_solvable_states(goal_state, num_states=3)
             if not belief_states:
                 raise ValueError("Belief Gen Error") 
             actual_start_state = next(iter(belief_states))
             tiles = [Tile(actual_start_state[i * 3 + j], j * 120 + 50, i * 120 + 50) for i in range(3) for j in range(3)]
             draw(tiles); pygame.event.pump() 
             print(f"Running {algo_name} with belief: {belief_states}")
             solution_path, solve_time = algo_func(belief_states, goal_state)
        elif algo_name == "Q-Learning":
            print(f"Running {algo_name} (Training may take time)...")
            solution_path, solve_time = algo_func(initial_state, goal_state)
        else:
             # Standard algorithms
             if not is_solvable(initial_state, goal_state):
                 raise ValueError("Not Solvable") 
             print(f"Running {algo_name} from {initial_state}...")
             solution_path, solve_time = algo_func(initial_state, goal_state)

    except ValueError as ve: # Catch specific errors we raise
        solution_path = str(ve)
        solve_time = 0.0
    except Exception as e:
        print(f"!!! Error during {algo_name} execution: {e}")
        import traceback
        traceback.print_exc()
        solution_path = f"Runtime Error"
        solve_time = 0.0

    post_solve_time = time.time() # Time after algorithm finishes

    # Update global display values
    current_result = solution_path
    solve_time_display = solve_time
    display_time_display = post_solve_time - pre_solve_time # Calculate display time

    print(f"{algo_name} Result: {solution_path} (Solve: {solve_time:.4f}s, Display: {display_time_display:.4f}s)")

    # Draw final status before animation
    draw(tiles)
    pygame.event.pump()

    # --- Animate the solution ---
    valid_moves = {'U', 'D', 'L', 'R'}
    is_valid_path = isinstance(solution_path, str) and solution_path and all(m in valid_moves for m in solution_path)

    if not is_valid_path:
        print("No valid path found or algorithm failed/terminated. No animation.")
        return

    animation_active = True
    clock = pygame.time.Clock()
    current_visual_state = list(actual_start_state) # Use state shown before algo run
    animation_start_time = time.time()
    print(f"Animating path ({len(solution_path)} steps): {solution_path[:60]}...") # Print start of path

    for move_index, move in enumerate(solution_path):
        if not animation_active: print("Animation interrupted."); break

        # Event handling during animation
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                 mouse_pos = e.pos
                 for btn_name, rect in buttons.items():
                     if rect.collidepoint(mouse_pos):
                         print(f"Animation interrupted by clicking {btn_name}")
                         animation_active = False
                         handle_button_click(btn_name) # Start new run
                         return # Exit animation

        # --- Perform animation step ---
        try: zero_idx = current_visual_state.index(0)
        except ValueError: print("Anim Error: Blank lost"); animation_active = False; break
        r, c = zero_idx // 3, zero_idx % 3
        target_swap_idx = -1
        if move == 'U' and r > 0: target_swap_idx = zero_idx - 3
        elif move == 'D' and r < 2: target_swap_idx = zero_idx + 3
        elif move == 'L' and c > 0: target_swap_idx = zero_idx - 1
        elif move == 'R' and c < 2: target_swap_idx = zero_idx + 1

        if target_swap_idx == -1: print(f"Anim Warning: Invalid move '{move}'"); continue

        try:
            blank_tile_obj = next(t for t in tiles if t.value == 0)
            moving_tile_value = current_visual_state[target_swap_idx]
            if moving_tile_value == 0: print(f"Anim Error: Swap blank?"); continue
            moving_tile_obj = next(t for t in tiles if t.value == moving_tile_value)
        except StopIteration: print(f"Anim Error: Tile not found"); animation_active = False; break

        # Set targets
        blank_r, blank_c = zero_idx // 3, zero_idx % 3
        moving_r, moving_c = target_swap_idx // 3, target_swap_idx % 3
        blank_tile_obj.target_x, blank_tile_obj.target_y = moving_c*120+50, moving_r*120+50
        moving_tile_obj.target_x, moving_tile_obj.target_y = blank_c*120+50, blank_r*120+50

        # Animation loop for one step
        while animation_active and (blank_tile_obj.is_moving() or moving_tile_obj.is_moving()):
            for e in pygame.event.get(): # Inner event loop for responsiveness
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                if e.type == pygame.MOUSEBUTTONDOWN:
                     mouse_pos = e.pos
                     for btn_name, rect in buttons.items():
                         if rect.collidepoint(mouse_pos):
                             print(f"Animation interrupted mid-step by clicking {btn_name}")
                             animation_active = False; handle_button_click(btn_name); return

            if not animation_active: break
            blank_tile_obj.move()
            moving_tile_obj.move()
            draw(tiles)
            clock.tick(60) 
        if not animation_active: break 

        
        current_visual_state[zero_idx], current_visual_state[target_swap_idx] = \
            current_visual_state[target_swap_idx], current_visual_state[zero_idx]
        blank_tile_obj.rect.topleft = (blank_tile_obj.target_x, blank_tile_obj.target_y)
        moving_tile_obj.rect.topleft = (moving_tile_obj.target_x, moving_tile_obj.target_y) 

    # --- Animation Finished ---
    animation_active = False
    animation_total_time = time.time() - animation_start_time
    if is_valid_path: print(f"Animation finished in {animation_total_time:.2f}s")

    # Final draw to show correct state
    final_state_to_draw = tuple(current_visual_state)
    tiles = [Tile(final_state_to_draw[i * 3 + j], j * 120 + 50, i * 120 + 50) for i in range(3) for j in range(3)]
    draw(tiles)


def handle_button_click(btn_name):
    """Handles the logic when an algorithm button is clicked."""
    print(f"--- Button Clicked: {btn_name} ---")
    # Use lambdas for algorithms needing specific parameters from your original mapping
    algo_map = {
        "BFS": bfs,
        "DFS": lambda start, goal: dfs(start, goal, max_depth=35),
        "IDDFS": lambda start, goal: iddfs(start, goal, max_limit=50),
        "UCS": ucs,
        "Greedy": greedy,
        "A*": astar,
        "IDA*": ida_star,
        "Hill": hill_climbing, 
        "Steepest": steepest_ascent_hill_climbing, 
        "Stochastic": lambda start, goal: stochastic_hill_climbing(start, goal, max_iterations=2000), 
        "Beam": lambda start, goal: beam_search(start, goal, beam_width=5), 
        "Sim. Anneal": lambda start, goal: simulated_annealing(start, goal, initial_temp=100, cooling_rate=0.99), 
        "Genetic": lambda start, goal: genetic_algorithm(start, goal, population_size=100, max_generations=200, mutation_rate=0.15, path_length=50),
        "Backtrack": lambda start, goal: backtracking(start, goal, max_depth=35),
        "CSP BT": lambda start, goal: csp_backtracking(start, goal, max_depth=35),
        "Q-Learning": lambda start, goal: q_learning(start, goal, episodes=20000, alpha=0.1, gamma=0.9, epsilon=0.2, max_steps_per_episode=150),
        "Sensorless": sensorless_bfs, 
        "Sensor BFS": sensorless_bfs_sensor, 
    }

    canonical_name_map = {"Sim. Anneal": "Simulated Annealing"}
    display_name = btn_name
    canonical_name = canonical_name_map.get(btn_name, btn_name)

    if display_name in algo_map:
        # Pass the global start 'd' and goal 'g'
        run_algorithm(algo_map[display_name], canonical_name, initial_state=d, goal_state=g)
    else:
        print(f"Error: Button '{display_name}' not mapped.")
        global current_algo, current_result
        current_algo = display_name
        current_result = "Algo Error (Mapping)"
        draw(tiles)


def main():
    """Main game loop."""
    global tiles 

    # Initial setup
    tiles = [Tile(d[i * 3 + j], j * 120 + 50, i * 120 + 50) for i in range(3) for j in range(3)]
    clock = pygame.time.Clock()

    # Initial draw
    if not is_solvable(d, g):
        print(f"WARNING: Initial state {d} is not solvable for goal {g}!")
        global current_result
        current_result = "Start State Unsolvable!"

    draw(tiles)

    while True:
        # Event handling
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1: 
                    if not animation_active:
                        mouse_pos = e.pos
                        for btn_name, rect in buttons.items():
                            if rect.collidepoint(mouse_pos):
                                handle_button_click(btn_name)
                                break 

        if not animation_active:
             draw(tiles) 

        clock.tick(30) 


# --- Main Execution ---
if __name__ == "__main__":
    main()