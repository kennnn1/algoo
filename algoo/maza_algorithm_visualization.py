import pygame 
import sys
import random
import math
import time
from collections import deque
import heapq

pygame.init()

SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
CELL_SIZE = 20
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
PURPLE = (128, 0, 128)
LIGHT_BLUE = (173, 216, 230)
LIGHT_PINK = (255, 182, 193)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)

# Application states
MAIN_MENU = 0
ALGORITHM_SELECTION = 1
ALGORITHM_VISUALIZATION = 2
MAZE_GAME = 3
INSTRUCTIONS = 4

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return self.x == other.x and self.y == other.y
        return False
    
    def __hash__(self):
        return hash((self.x, self.y))
    
    def __str__(self):
        return f"({self.x}, {self.y})"

class Cell:
    WALL = 1
    PATH = 0
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.is_wall = True
        self.is_visited = False
        self.is_path = False
        self.is_explored = False
        self.is_backtracked = False
        self.is_current = False
    
    def reset(self):
        self.is_visited = False
        self.is_path = False
        self.is_explored = False
        self.is_backtracked = False
        self.is_current = False

class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = [[Cell(x, y) for y in range(height)] for x in range(width)]
        self.start_x = 1
        self.start_y = 1
        self.end_x = width - 2
        self.end_y = height - 2
        self.generate_maze()
    
    def generate_maze(self):
    
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x][y].is_wall = True
        
   
        for x in range(self.width):
            self.grid[x][0].is_wall = True
            self.grid[x][self.height-1].is_wall = True
        for y in range(self.height):
            self.grid[0][y].is_wall = True
            self.grid[self.width-1][y].is_wall = True
        
    
        self.grid[self.start_x][self.start_y].is_wall = False
        self._generate_paths(self.start_x, self.start_y)
        

        self.grid[self.end_x][self.end_y].is_wall = False
        
        self._add_random_paths()
    
    def _generate_paths(self, x, y):
        directions = [(0, -2), (2, 0), (0, 2), (-2, 0)]  # Up, right, down, left
        random.shuffle(directions)
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 < nx < self.width-1 and 0 < ny < self.height-1 and 
                self.grid[nx][ny].is_wall):
                
              
                self.grid[nx][ny].is_wall = False
                self.grid[x + dx//2][y + dy//2].is_wall = False
                
     
                self._generate_paths(nx, ny)
    
    def _add_random_paths(self):
        for _ in range(self.width * self.height // 10):
            x = random.randint(1, self.width - 2)
            y = random.randint(1, self.height - 2)
            if self.grid[x][y].is_wall:
                self.grid[x][y].is_wall = False
    
    def reset_path_markers(self):
        for x in range(self.width):
            for y in range(self.height):
                self.grid[x][y].reset()
    
    def is_wall(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y].is_wall
        return True
    
    def set_path(self, x, y, is_path):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[x][y].is_path = is_path
    
    def convert_to_grid(self):
        """Convert maze to a simple 2D grid of 0s and 1s for algorithm use"""
        grid = [[0 for _ in range(self.height)] for _ in range(self.width)]
        for x in range(self.width):
            for y in range(self.height):
                grid[x][y] = 1 if self.grid[x][y].is_wall else 0
        return grid

class PathfindingAlgorithm:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def find_path(self, maze, start_x, start_y, end_x, end_y, callback=None):
        """Find a path from start to end in the maze"""
        pass

class BacktrackingAlgorithm(PathfindingAlgorithm):
    def __init__(self):
        super().__init__(
            "Backtracking Algorithm", 
            "A recursive depth-first search approach that explores as far as possible along a branch before backtracking."
        )
    
    def find_path(self, maze, start_x, start_y, end_x, end_y, callback=None):
        grid = maze.convert_to_grid()
        height = len(grid[0])
        width = len(grid)
        visited = [[False for _ in range(height)] for _ in range(width)]
        path = []
        
        def find_path_recursive(x, y):
            if (x < 0 or y < 0 or x >= width or y >= height or 
                grid[x][y] == 1 or visited[y][x]):
                return False
            
            current = Point(x, y)
            
            if callback:
                callback(current, "current")
            
            path.append(current)
            visited[y][x] = True
            
            if callback:
                callback(current, "explore")
            
            if x == end_x and y == end_y:
                if callback:
                    for p in path:
                        callback(p, "path")
                return True
            
            # (up, right, down, left)
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            
            for dx, dy in directions:
                if find_path_recursive(x + dx, y + dy):
                    return True
            
            path.pop()
            
            if callback:
                callback(current, "backtrack")
            
            return False
        
        find_path_recursive(start_x, start_y)
        return path

class GreedyAlgorithm(PathfindingAlgorithm):
    def __init__(self):
        super().__init__(
            "Greedy Algorithm",
            "Makes locally optimal choices based on Manhattan distance to the goal, attempting to find a solution quickly."
        )
    
    def find_path(self, maze, start_x, start_y, end_x, end_y, callback=None):
        grid = maze.convert_to_grid()
        height = len(grid[0])
        width = len(grid)
        
        queue = []
        visited = [[False for _ in range(height)] for _ in range(width)]
        parent_map = {}  
        
        def manhattan_distance(x1, y1, x2, y2):
            return abs(x1 - x2) + abs(y1 - y2)
        
        start_dist = manhattan_distance(start_x, start_y, end_x, end_y)
        heapq.heappush(queue, (start_dist, start_x, start_y))
        
        while queue:
            _, x, y = heapq.heappop(queue)
            current = Point(x, y)
            
            if callback:
                callback(current, "current")
            
            if visited[y][x]:
                continue
            
            visited[y][x] = True
            
            if callback:
                callback(current, "explore")
            
            if x == end_x and y == end_y:
                path = []
                curr_x, curr_y = end_x, end_y
                while (curr_x, curr_y) in parent_map:
                    point = Point(curr_x, curr_y)
                    path.append(point)
                    curr_x, curr_y = parent_map[(curr_x, curr_y)]
                
                path.append(Point(start_x, start_y))
                
                path.reverse()
                
                if callback:
                    for p in path:
                        callback(p, "path")
                
                return path
            
            # (up, right, down, left)
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                
                if (nx < 0 or ny < 0 or nx >= width or ny >= height or 
                    grid[nx][ny] == 1 or visited[ny][nx]):
                    continue
                
                priority = manhattan_distance(nx, ny, end_x, end_y)
                
                
                heapq.heappush(queue, (priority, nx, ny))
                
                if (nx, ny) not in parent_map:
                    parent_map[(nx, ny)] = (x, y)
        
        
        return []

class JumpSearchAlgorithm(PathfindingAlgorithm):
    def __init__(self):
        super().__init__(
            "Jump Search Algorithm",
            "Jumps in steps of sqrt(n) where n is the maze dimension, reducing the number of cells examined compared to linear search."
        )
    
    def find_path(self, maze, start_x, start_y, end_x, end_y, callback=None):
        grid = maze.convert_to_grid()
        height = len(grid[0])
        width = len(grid)
        
        jump_size = int(math.sqrt(max(width, height)))
        
        queue = deque([(start_x, start_y)])
        visited = [[False for _ in range(height)] for _ in range(width)]
        visited[start_y][start_x] = True
        
        parent_map = {}
        
        while queue:
            x, y = queue.popleft()
            current = Point(x, y)
            
            if callback:
                callback(current, "current")
            
            if x == end_x and y == end_y:
             
                raw_path = []
                curr_x, curr_y = end_x, end_y
                while (curr_x, curr_y) in parent_map:
                    raw_path.append(Point(curr_x, curr_y))
                    curr_x, curr_y = parent_map[(curr_x, curr_y)]
                
                
                raw_path.append(Point(start_x, start_y))
                
            
                raw_path.reverse()
            
                path = []
                for i in range(len(raw_path) - 1):
                    p1 = raw_path[i]
                    p2 = raw_path[i + 1]
                   
                    path.append(p1)
                    
              
                    self._fill_path_between_points(p1, p2, path)
                
          
                if not any(p.x == end_x and p.y == end_y for p in path):
                    path.append(Point(end_x, end_y))
                
            
                if callback:
                    for p in path:
                        callback(p, "path")
                
                return path
            
       
            if callback:
                callback(current, "explore")
          
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # Up, right, down, left
            
            for dx, dy in directions:
                self._jump_in_direction(grid, x, y, dx, dy, jump_size, visited, queue, parent_map, callback)
        
      
        return []
    
    def _jump_in_direction(self, grid, start_x, start_y, dx, dy, jump_size, visited, queue, parent_map, callback):
        width = len(grid)
        height = len(grid[0])
        x, y = start_x, start_y
        
        for i in range(1, jump_size + 1):
            x += dx
            y += dy
            
            if x < 0 or y < 0 or x >= width or y >= height:
                break
            
            if grid[x][y] == 1:
                break
            
            if visited[y][x]:
                continue
            
            visited[y][x] = True
            
            if callback:
                callback(Point(x, y), "explore")
            
            queue.append((x, y))
            
            parent_map[(x, y)] = (start_x, start_y)
    
    def _fill_path_between_points(self, p1, p2, path):
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        while not (x1 == x2 and y1 == y2):
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
            
            if not (x1 == x2 and y1 == y2):
                path.append(Point(x1, y1))

class AlgorithmVisualizer:
    def __init__(self, maze, screen):
        self.maze = maze
        self.screen = screen
        self.cell_size = CELL_SIZE
        self.font = pygame.font.SysFont('arial', 14)
        self.title_font = pygame.font.SysFont('arial', 24)
        self.visualization_steps = []
        self.current_step = 0
        self.is_playing = False
        self.play_speed = 10  
        self.last_step_time = 0
        self.final_path = []
        self.visualization_complete = False
    
    def visualize_algorithm(self, algorithm):
        self.maze.reset_path_markers()
        
        self.visualization_steps = []
        self.current_step = 0
        self.is_playing = False
        self.final_path = []
        self.visualization_complete = False
        
        def callback(point, step_type):
            self.visualization_steps.append((point, step_type))
            
            x, y = point.x, point.y
            if 0 <= x < self.maze.width and 0 <= y < self.maze.height and not self.maze.grid[x][y].is_wall:
                cell = self.maze.grid[x][y]
                
                if step_type == "explore":
                    cell.is_explored = True
                elif step_type == "backtrack":
                    cell.is_backtracked = True
                elif step_type == "path":
                    cell.is_path = True
                    if point not in self.final_path:
                        self.final_path.append(point)
                elif step_type == "current":
                    cell.is_current = True
        
        path = algorithm.find_path(
            self.maze, 
            self.maze.start_x, 
            self.maze.start_y, 
            self.maze.end_x, 
            self.maze.end_y, 
            callback
        )
        
        self.final_path = path
        
        self.is_playing = True
        self.last_step_time = pygame.time.get_ticks()
    
    def toggle_play_pause(self):
        self.is_playing = not self.is_playing
        self.last_step_time = pygame.time.get_ticks()
    
    def next_step(self):
        if self.current_step < len(self.visualization_steps):
            point, step_type = self.visualization_steps[self.current_step]
            self.update_maze_state(point, step_type)
            self.current_step += 1
            
            if self.current_step >= len(self.visualization_steps):
                self.visualization_complete = True
    
    def update_maze_state(self, point, step_type):
        x, y = point.x, point.y
        
        if 0 <= x < self.maze.width and 0 <= y < self.maze.height and not self.maze.grid[x][y].is_wall:
            cell = self.maze.grid[x][y]
            
            if step_type == "explore":
                cell.is_explored = True
                cell.is_current = False
            elif step_type == "backtrack":
                cell.is_backtracked = True
                cell.is_explored = False
            elif step_type == "path":
                cell.is_path = True
            elif step_type == "current":
                cell.is_current = True
    
    def skip_to_end(self):
        for point, step_type in self.visualization_steps[self.current_step:]:
            self.update_maze_state(point, step_type)
        
        self.current_step = len(self.visualization_steps)
        self.visualization_complete = True
    
    def update(self):
        if self.is_playing and not self.visualization_complete:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_step_time > 1000 / self.play_speed:
                self.next_step()
                self.last_step_time = current_time
    
    def change_speed(self, delta):
        self.play_speed = max(1, min(30, self.play_speed + delta))
    
    def draw(self):
        for x in range(self.maze.width):
            for y in range(self.maze.height):
                cell = self.maze.grid[x][y]
                rect = pygame.Rect(
                    x * self.cell_size, 
                    y * self.cell_size, 
                    self.cell_size, 
                    self.cell_size
                )
                
                if cell.is_wall:
                    pygame.draw.rect(self.screen, BLACK, rect)
                elif (x, y) == (self.maze.start_x, self.maze.start_y):
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif (x, y) == (self.maze.end_x, self.maze.end_y):
                    pygame.draw.rect(self.screen, ORANGE, rect)
                elif cell.is_path:
                    pygame.draw.rect(self.screen, YELLOW, rect)
                elif cell.is_current:
                    pygame.draw.rect(self.screen, RED, rect)
                elif cell.is_backtracked:
                    pygame.draw.rect(self.screen, LIGHT_PINK, rect)
                elif cell.is_explored:
                    pygame.draw.rect(self.screen, LIGHT_BLUE, rect)
                else:
                    pygame.draw.rect(self.screen, WHITE, rect)
                
                pygame.draw.rect(self.screen, GRAY, rect, 1)
        
        title_text = self.title_font.render("Algorithm Visualization", True, BLACK)
        self.screen.blit(title_text, (self.maze.width * self.cell_size + 20, 20))
        
        step_text = self.font.render(
            f"Step: {self.current_step}/{len(self.visualization_steps)}", 
            True, 
            BLACK
        )
        self.screen.blit(step_text, (self.maze.width * self.cell_size + 20, 60))
        
        # Controls
        controls_y = 100
        controls_text = [
            "Controls:",
            "Space - Play/Pause",
            "Right Arrow - Next Step",
            "End - Skip to End",
            "+ / - - Change Speed",
            "Enter - Start Game (when complete)"
        ]
        
        for text in controls_text:
            text_surface = self.font.render(text, True, BLACK)
            self.screen.blit(text_surface, (self.maze.width * self.cell_size + 20, controls_y))
            controls_y += 25
        
     
        legend_y = 250
        legend_text = self.font.render("Legend:", True, BLACK)
        self.screen.blit(legend_text, (self.maze.width * self.cell_size + 20, legend_y))
        
        legend_items = [
            (WHITE, "Unexplored"),
            (LIGHT_BLUE, "Explored"),
            (LIGHT_PINK, "Backtracked"),
            (YELLOW, "Path"),
            (RED, "Current"),
            (GREEN, "Start"),
            (ORANGE, "End"),
            (BLACK, "Wall")
        ]
        
        for i, (color, text) in enumerate(legend_items):
            y_pos = legend_y + 30 + i * 25
            
            pygame.draw.rect(
                self.screen, 
                color, 
                pygame.Rect(self.maze.width * self.cell_size + 20, y_pos, 15, 15)
            )
            pygame.draw.rect(
                self.screen, 
                BLACK, 
                pygame.Rect(self.maze.width * self.cell_size + 20, y_pos, 15, 15), 
                1
            )
            
            text_surface = self.font.render(text, True, BLACK)
            self.screen.blit(text_surface, (self.maze.width * self.cell_size + 45, y_pos))
        
        status_y = 480
        if self.visualization_complete:
            status_text = self.font.render(
                f"Visualization complete! Path found with {len(self.final_path)} steps.", 
                True, 
                GREEN
            )
            self.screen.blit(status_text, (self.maze.width * self.cell_size + 20, status_y))
            
            start_text = self.font.render(
                "Press ENTER to start the maze game", 
                True, 
                BLUE
            )
            self.screen.blit(start_text, (self.maze.width * self.cell_size + 20, status_y + 30))
        else:
            if self.is_playing:
                status_text = self.font.render("Visualizing... (Press SPACE to pause)", True, BLUE)
            else:
                status_text = self.font.render("Paused (Press SPACE to continue)", True, RED)
            
            self.screen.blit(status_text, (self.maze.width * self.cell_size + 20, status_y))
        
        speed_text = self.font.render(f"Speed: {self.play_speed}x", True, BLACK)
        self.screen.blit(speed_text, (self.maze.width * self.cell_size + 20, status_y + 60))

class Player:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.direction_x = 1.0
        self.direction_y = 0.0
        self.plane_x = 0.0
        self.plane_y = 0.66
        self.move_speed = 0.05
        self.rotation_speed = 0.03
    
    def move_forward(self, maze):
        new_x = self.x + self.direction_x * self.move_speed
        new_y = self.y + self.direction_y * self.move_speed
        self._try_move(maze, new_x, new_y)
    
    def move_backward(self, maze):
        new_x = self.x - self.direction_x * self.move_speed
        new_y = self.y - self.direction_y * self.move_speed
        self._try_move(maze, new_x, new_y)
    
    def strafe_left(self, maze):
        new_x = self.x + self.direction_y * self.move_speed
        new_y = self.y - self.direction_x * self.move_speed
        self._try_move(maze, new_x, new_y)
    
    def strafe_right(self, maze):
        new_x = self.x - self.direction_y * self.move_speed
        new_y = self.y + self.direction_x * self.move_speed
        self._try_move(maze, new_x, new_y)
    
    def rotate_right(self):
        # Rotate clockwise (right)
        old_dir_x = self.direction_x
        self.direction_x = self.direction_x * math.cos(self.rotation_speed) - self.direction_y * math.sin(self.rotation_speed)
        self.direction_y = old_dir_x * math.sin(self.rotation_speed) + self.direction_y * math.cos(self.rotation_speed)
        old_plane_x = self.plane_x
        self.plane_x = self.plane_x * math.cos(self.rotation_speed) - self.plane_y * math.sin(self.rotation_speed)
        self.plane_y = old_plane_x * math.sin(self.rotation_speed) + self.plane_y * math.cos(self.rotation_speed)
    
    def rotate_left(self):
        # Rotate counter-clockwise (left)
        old_dir_x = self.direction_x
        self.direction_x = self.direction_x * math.cos(-self.rotation_speed) - self.direction_y * math.sin(-self.rotation_speed)
        self.direction_y = old_dir_x * math.sin(-self.rotation_speed) + self.direction_y * math.cos(-self.rotation_speed)
        old_plane_x = self.plane_x
        self.plane_x = self.plane_x * math.cos(-self.rotation_speed) - self.plane_y * math.sin(-self.rotation_speed)
        self.plane_y = old_plane_x * math.sin(-self.rotation_speed) + self.plane_y * math.cos(-self.rotation_speed)
    
    def _try_move(self, maze, new_x, new_y):
        if not maze.is_wall(int(new_x), int(self.y)):
            self.x = new_x
        if not maze.is_wall(int(self.x), int(new_y)):
            self.y = new_y

class MazeGame:
    def __init__(self, screen, width=30, height=30):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.maze = Maze(width, height)
        self.state = MAIN_MENU
        self.player = Player(self.maze.start_x + 0.5, self.maze.start_y + 0.5)
        self.algorithm = None
        self.path = []
        self.cell_size = CELL_SIZE
        self.visualizer = None
        
        
        self.wall_height = 300
        self.screen_width = SCREEN_WIDTH
        self.screen_height = SCREEN_HEIGHT
        
        self.font = pygame.font.SysFont('arial', 20)
        self.title_font = pygame.font.SysFont('arial', 36)
        
        self.algorithms = [
            BacktrackingAlgorithm(),
            GreedyAlgorithm(),
            JumpSearchAlgorithm()
        ]
    
    def run(self):
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        if self.state == MAIN_MENU:
                            running = False
                        else:
                            self.state = MAIN_MENU
                    
                    self._handle_key_press(event)
            
            if self.state == MAIN_MENU:
                self._update_main_menu()
            elif self.state == ALGORITHM_SELECTION:
                self._update_algorithm_selection()
            elif self.state == ALGORITHM_VISUALIZATION:
                self._update_visualization()
            elif self.state == MAZE_GAME:
                self._update_game()
            elif self.state == INSTRUCTIONS:
                self._update_instructions()
            
            self.screen.fill(WHITE)
            
            if self.state == MAIN_MENU:
                self._draw_main_menu()
            elif self.state == ALGORITHM_SELECTION:
                self._draw_algorithm_selection()
            elif self.state == ALGORITHM_VISUALIZATION:
                self._draw_visualization()
            elif self.state == MAZE_GAME:
                self._draw_game()
            elif self.state == INSTRUCTIONS:
                self._draw_instructions()
            
            pygame.display.flip()
            self.clock.tick(FPS)
        
        pygame.quit()
    
    def _handle_key_press(self, event):
        if self.state == MAIN_MENU:
            if event.key == pygame.K_SPACE or event.key == pygame.K_RETURN:
                self.state = ALGORITHM_SELECTION
            elif event.key == pygame.K_i:
                self.state = INSTRUCTIONS
        
        elif self.state == ALGORITHM_SELECTION:
            if event.key == pygame.K_1 or event.key == pygame.K_KP1:
                self.algorithm = self.algorithms[0]
                self._start_visualization()
            elif event.key == pygame.K_2 or event.key == pygame.K_KP2:
                self.algorithm = self.algorithms[1]
                self._start_visualization()
            elif event.key == pygame.K_3 or event.key == pygame.K_KP3:
                self.algorithm = self.algorithms[2]
                self._start_visualization()
            elif event.key == pygame.K_ESCAPE:
                self.state = MAIN_MENU
        
        elif self.state == ALGORITHM_VISUALIZATION:
            if event.key == pygame.K_SPACE:
                self.visualizer.toggle_play_pause()
            elif event.key == pygame.K_RIGHT:
                self.visualizer.next_step()
            elif event.key == pygame.K_END:
                self.visualizer.skip_to_end()
            elif event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS or event.key == pygame.K_EQUALS:
                self.visualizer.change_speed(1)
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                self.visualizer.change_speed(-1)
            elif (event.key == pygame.K_RETURN and self.visualizer.visualization_complete):
                self._start_game()
        
        elif self.state == MAZE_GAME:
            pass
    
    def _update_main_menu(self):
        pass  
    
    def _update_algorithm_selection(self):
        pass  
    
    def _update_visualization(self):
        if self.visualizer:
            self.visualizer.update()

    def _update_game(self):
        keys = pygame.key.get_pressed()
        
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.player.move_forward(self.maze)
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.player.move_backward(self.maze)
        if keys[pygame.K_a]:
            self.player.strafe_left(self.maze)
        if keys[pygame.K_d]:
            self.player.strafe_right(self.maze)
        if keys[pygame.K_LEFT]:
            self.player.rotate_left()
        if keys[pygame.K_RIGHT]:
            self.player.rotate_right()
        
        player_x_int = int(self.player.x)
        player_y_int = int(self.player.y)
        
        if (player_x_int == self.maze.end_x and player_y_int == self.maze.end_y):
            font = pygame.font.SysFont('arial', 48)
            text = font.render("Maze Completed!", True, GREEN)
            self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, SCREEN_HEIGHT//2 - text.get_height()//2))
            pygame.display.flip()
            
            pygame.time.wait(3000)
            self.state = MAIN_MENU
    
    def _update_instructions(self):
        pass  
    
    def _draw_main_menu(self):
        title_text = self.title_font.render("Maze Algorithm Visualization", True, BLACK)
        self.screen.blit(title_text, (SCREEN_WIDTH//2 - title_text.get_width()//2, 100))
        
        instructions = [
            "Welcome to the Maze Algorithm Visualization!",
            "",
            "This application demonstrates how different pathfinding algorithms",
            "work to find a path through a maze.",
            "",
            "Press SPACE to select an algorithm",
            "Press I for instructions",
            "Press ESC to quit"
        ]
        
        y_pos = 200
        for line in instructions:
            text = self.font.render(line, True, BLACK)
            self.screen.blit(text, (SCREEN_WIDTH//2 - text.get_width()//2, y_pos))
            y_pos += 40
    
    def _draw_algorithm_selection(self):
        
        title_text = self.title_font.render("Select an Algorithm", True, BLACK)
        self.screen.blit(title_text, (SCREEN_WIDTH//2 - title_text.get_width()//2, 100))
        
        
        y_pos = 180
        for i, algorithm in enumerate(self.algorithms):
            
            number_text = self.font.render(f"{i+1}. ", True, BLACK)
            self.screen.blit(number_text, (SCREEN_WIDTH//4, y_pos))
            
            name_text = self.font.render(algorithm.name, True, BLUE)
            self.screen.blit(name_text, (SCREEN_WIDTH//4 + 30, y_pos))
            
            
            desc_lines = self._wrap_text(algorithm.description, 60)
            for line in desc_lines:
                y_pos += 30
                desc_text = self.font.render(line, True, BLACK)
                self.screen.blit(desc_text, (SCREEN_WIDTH//4 + 50, y_pos))
            
            y_pos += 60
        
        instructions_text = self.font.render("Press 1, 2, or 3 to select an algorithm. Press ESC to go back.", True, BLACK)
        self.screen.blit(instructions_text, (SCREEN_WIDTH//2 - instructions_text.get_width()//2, SCREEN_HEIGHT - 100))
    
    def _draw_visualization(self):
        if self.visualizer:
            self.visualizer.draw()
    
    def _draw_game(self):
        self.screen.fill(BLACK)
        
        pygame.draw.rect(self.screen, GRAY, (0, 0, self.screen_width, self.screen_height // 2))
        
        pygame.draw.rect(self.screen, DARK_GRAY, (0, self.screen_height // 2, self.screen_width, self.screen_height // 2))
        
        for x in range(self.screen_width):
            # calculate ray position and direction
            camera_x = 2 * x / self.screen_width - 1
            ray_dir_x = self.player.direction_x + self.player.plane_x * camera_x
            ray_dir_y = self.player.direction_y + self.player.plane_y * camera_x
            
            # current map position
            map_x = int(self.player.x)
            map_y = int(self.player.y)
            
            # length of ray from current position to next x or y-side
            delta_dist_x = float('inf') if ray_dir_x == 0 else abs(1 / ray_dir_x)
            delta_dist_y = float('inf') if ray_dir_y == 0 else abs(1 / ray_dir_y)
            
            # direction to step in x or y direction
            step_x = 1 if ray_dir_x >= 0 else -1
            step_y = 1 if ray_dir_y >= 0 else -1
            
            # length of ray from current position to next x or y-side
            if ray_dir_x < 0:
                side_dist_x = (self.player.x - map_x) * delta_dist_x
            else:
                side_dist_x = (map_x + 1.0 - self.player.x) * delta_dist_x
            
            if ray_dir_y < 0:
                side_dist_y = (self.player.y - map_y) * delta_dist_y
            else:
                side_dist_y = (map_y + 1.0 - self.player.y) * delta_dist_y
            
            hit = False
            side = 0  
            
            while not hit:
          
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1
                
  
                if self.maze.is_wall(map_x, map_y):
                    hit = True
            
   
            if side == 0:
                perp_wall_dist = (map_x - self.player.x + (1 - step_x) / 2) / ray_dir_x
            else:
                perp_wall_dist = (map_y - self.player.y + (1 - step_y) / 2) / ray_dir_y
            
            line_height = int(self.screen_height / perp_wall_dist) if perp_wall_dist > 0 else self.screen_height
            
            draw_start = -line_height // 2 + self.screen_height // 2
            if draw_start < 0:
                draw_start = 0
            
            draw_end = line_height // 2 + self.screen_height // 2
            if draw_end >= self.screen_height:
                draw_end = self.screen_height - 1
            
            wall_color = BLUE
            
            is_path_wall = False
            for point in self.path:
                if point.x == map_x and point.y == map_y:
                    is_path_wall = True
                    break
            
            if is_path_wall:
                wall_color = YELLOW  
            elif side == 1:
                wall_color = (0, 0, 160)  
            
            pygame.draw.line(self.screen, wall_color, (x, draw_start), (x, draw_end), 1)
        
        self._draw_minimap()
        
        self._draw_path_indicator()
    
    def _draw_minimap(self):
        minimap_size = 5 
        minimap_width = self.maze.width * minimap_size
        minimap_height = self.maze.height * minimap_size
        
        pygame.draw.rect(self.screen, (0, 0, 0, 128), 
                         (SCREEN_WIDTH - minimap_width - 10, 10, minimap_width, minimap_height))
        
        for x in range(self.maze.width):
            for y in range(self.maze.height):
                rect = pygame.Rect(
                    SCREEN_WIDTH - minimap_width - 10 + x * minimap_size,
                    10 + y * minimap_size,
                    minimap_size,
                    minimap_size
                )
                
                if self.maze.is_wall(x, y):
                    pygame.draw.rect(self.screen, BLACK, rect)
                elif (x, y) == (self.maze.start_x, self.maze.start_y):
                    pygame.draw.rect(self.screen, GREEN, rect)
                elif (x, y) == (self.maze.end_x, self.maze.end_y):
                    pygame.draw.rect(self.screen, RED, rect)
                else:
                    is_path = False
                    for point in self.path:
                        if point.x == x and point.y == y:
                            is_path = True
                            break
                    
                    if is_path:
                        pygame.draw.rect(self.screen, YELLOW, rect)
                    else:
                        pygame.draw.rect(self.screen, WHITE, rect)
        
        player_x = SCREEN_WIDTH - minimap_width - 10 + int(self.player.x * minimap_size)
        player_y = 10 + int(self.player.y * minimap_size)
        pygame.draw.circle(self.screen, BLUE, (player_x, player_y), minimap_size // 2)
        
        dir_x = player_x + int(self.player.direction_x * minimap_size)
        dir_y = player_y + int(self.player.direction_y * minimap_size)
        pygame.draw.line(self.screen, BLUE, (player_x, player_y), (dir_x, dir_y), 1)
    
    def _draw_path_indicator(self):
        current_point = Point(int(self.player.x), int(self.player.y))
        
        next_point = None
        for i, point in enumerate(self.path):
            if point.x == current_point.x and point.y == current_point.y:
                if i + 1 < len(self.path):
                    next_point = self.path[i + 1]
                break
        
        if next_point:
            text = self.font.render("→ Next", True, YELLOW)
            self.screen.blit(text, (20, 20))
            
            minimap_size = 5
            minimap_width = self.maze.width * minimap_size
            
            next_x = SCREEN_WIDTH - minimap_width - 10 + int(next_point.x * minimap_size)
            next_y = 10 + int(next_point.y * minimap_size)
            
            pygame.draw.circle(self.screen, (255, 165, 0), (next_x, next_y), minimap_size // 2 + 1)
    
    def _draw_instructions(self):
        
        title_text = self.title_font.render("Instructions", True, BLACK)
        self.screen.blit(title_text, (SCREEN_WIDTH//2 - title_text.get_width()//2, 50))
        
        
        instructions = [
            "How to use this application:",
            "",
            "1. Select an algorithm from the menu",
            "2. Watch as the algorithm searches for a path",
            "3. Use the controls to play/pause or adjust the visualization speed",
            "4. When the visualization is complete, press ENTER to play the maze",
            "5. In the game, use the following controls:",
            "   - W/Up Arrow: Move forward",
            "   - S/Down Arrow: Move backward",
            "   - A: Strafe left",
            "   - D: Strafe right",
            "   - Left/Right Arrow: Rotate view",
            "",
            "The different algorithms work as follows:",
            "",
            "- Backtracking: Explores paths deeply before backtracking from dead ends",
            "- Greedy: Prioritizes moves that get closer to the goal",
            "- Jump Search: Takes larger jumps to explore the maze more efficiently",
            "",
            "Press ESC to return to the main menu"
        ]
        
        y_pos = 120
        for line in instructions:
            text = self.font.render(line, True, BLACK)
            self.screen.blit(text, (50, y_pos))
            y_pos += 30
    
    def _start_visualization(self):
        self.state = ALGORITHM_VISUALIZATION
        self.visualizer = AlgorithmVisualizer(self.maze, self.screen)
        self.visualizer.visualize_algorithm(self.algorithm)
    
    def _start_game(self):
        self.state = MAZE_GAME
        
        self.path = self.visualizer.final_path
        
        self.player = Player(self.maze.start_x + 0.5, self.maze.start_y + 0.5)
    
    def _wrap_text(self, text, max_chars_per_line):
        """Wrap text to fit within a certain width"""
        words = text.split(' ')
        lines = []
        current_line = ""
        
        for word in words:
            if len(current_line) + len(word) + 1 <= max_chars_per_line:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        
        if current_line:
            lines.append(current_line)
        
        return lines


def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Maze Algorithm Visualization")
    
    game = MazeGame(screen, width=25, height=25)
    game.run()


if __name__ == "__main__":
    main()

    