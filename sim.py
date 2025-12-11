import pygame
import numpy as np
import heapq

# --- SETTINGS ---
CELL = 60
MAZE_SIZE = 10
WIDTH, HEIGHT = MAZE_SIZE * CELL, MAZE_SIZE * CELL
FPS = 40

# --- CONSTANTS ---
STEP_COST = 1000
TURN_PENALTY = 5000 
L_DIST, L_DIR, L_WALL = 0, 1, 2
WALL_N, WALL_E, WALL_S, WALL_W = 1, 2, 4, 8

# --- COLORS ---
BG_COLOR = (10, 15, 20)
WALL_COLOR = (0, 255, 255)
PATH_COLOR = (255, 0, 255)
ROBOT_COLOR = (255, 255, 0)
VISITED_COLOR = (20, 40, 60)
GRID_COLOR = (25, 30, 45)

def get_maze_layout():
    # 10x10 Complex Room Maze
    return np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 1, 0, 0, 0, 0, 0, 1], 
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
        [1, 0, 0, 0, 0, 0, 0, 1, 0, 1], 
        [1, 1, 1, 0, 1, 1, 0, 1, 0, 1], 
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        [1, 0, 1, 1, 1, 1, 1, 1, 0, 1], 
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 1], 
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])

class Robot:
    def __init__(self, start):
        self.x, self.y = start
        self.dir = (1, 0)

        # 3D Memory: [Distance, Direction, Walls]
        self.mem = np.zeros((MAZE_SIZE, MAZE_SIZE, 3), dtype=int)
        self.mem[:, :, L_WALL] = 15 # Init as Unknown

        self.visited = set()
        self.visited.add((self.x, self.y))
        
        # Initialize start cell
        self.mem[self.y][self.x][L_DIST] = 0
        self.mem[self.y][self.x][L_WALL] = 0 # Assume open start for this demo

        self.stack = []
        self.finished = False
        self.final_path = []

    # ==========================================================
    #  THE "BRAIN" FUNCTION (Port this logic to Arduino)
    # ==========================================================
    def explore(self, ticks_measured, current_pos, detected_walls):
        """
        Takes raw sensor data and updates the internal map.
        
        Args:
            ticks_measured (int): Value from encoder since last stop.
            current_pos (tuple): (x, y) grid coordinates.
            detected_walls (int): Bitmask of walls sensed at current pos.
        """
        cx, cy = current_pos
        
        # 1. UPDATE WALLS (Topology Layer)
        # We perform a bitwise AND to remove walls that don't exist.
        # Ideally, sensors tell us what IS there. Here we use a negative mask logic
        # compatible with our previous "Remove Wall" approach.
        
        # If we arrived here, the path behind us is definitely open.
        # (This logic is handled in the move loop in simulation, 
        # but on Arduino, you'd update based on IR sensors here).
        
        # In this simulation, 'detected_walls' is the Truth from the map.
        # We overwrite our memory with what we see.
        self.mem[cy][cx][L_WALL] = detected_walls

        # 2. UPDATE DISTANCE (Cost Layer)
        # We only record the cost if this is the first visit.
        if self.mem[cy][cx][L_DIST] == 0 and (cx, cy) != (1,1):
            self.mem[cy][cx][L_DIST] = int(ticks_measured)

        # 3. MARK VISITED
        self.visited.add((cx, cy))

    # ==========================================================
    #  THE "HARDWARE" SIMULATION (Main Loop)
    # ==========================================================
    def move_hardware_simulation(self, real_maze):
        if self.finished: return

        cx, cy = self.x, self.y
        
        # --- 1. SENSE SURROUNDINGS (Simulating Sensors) ---
        # In real life, you would read IR sensors here.
        # Here, we peek at the 'real_maze' array.
        valid_neighbors = []
        
        # Check Right, Down, Left, Up
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1)]:
            nx, ny = cx+dx, cy+dy
            if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE:
                # If path is open (0)
                if real_maze[ny][nx] == 0: 
                    # If not visited, it's a candidate for exploration
                    if (nx,ny) not in self.visited:
                        valid_neighbors.append((nx, ny))

        # --- 2. DECIDE MOVEMENT (DFS Logic) ---
        next_pos = None
        if valid_neighbors:
            next_pos = valid_neighbors[0]
            self.stack.append((cx, cy))
        elif self.stack:
            next_pos = self.stack.pop()
        else:
            self.finished = True
            self.solve_grid_astar((1,1), (8,8))
            return

        # --- 3. EXECUTE MOVE (Simulate Motors) ---
        # Robot drives to next_pos...
        # ...
        # Robot Arrives.
        
        # --- 4. SIMULATE ENCODER READING ---
        # Encoder says: "I spun 1000 ticks to get here"
        simulated_encoder_value = STEP_COST 
        
        # --- 5. SIMULATE WALL SENSORS ---
        # To make the map accurate, we need to calculate the wall mask
        # for the NEW position we just arrived at.
        nx, ny = next_pos
        wall_mask = 0
        if ny > 0 and real_maze[ny-1][nx] == 1: wall_mask += WALL_N
        if nx < MAZE_SIZE-1 and real_maze[ny][nx+1] == 1: wall_mask += WALL_E
        if ny < MAZE_SIZE-1 and real_maze[ny+1][nx] == 1: wall_mask += WALL_S
        if nx > 0 and real_maze[ny][nx-1] == 1: wall_mask += WALL_W
        
        # We must also ensure the path we just came from is OPEN in the mask
        dx = nx - cx
        dy = ny - cy
        if dx == 1: wall_mask &= ~WALL_W # Came from West, so West is open
        elif dx == -1: wall_mask &= ~WALL_E
        elif dy == 1: wall_mask &= ~WALL_N
        elif dy == -1: wall_mask &= ~WALL_S

        # --- 6. CALL THE BRAIN FUNCTION ---
        self.explore(simulated_encoder_value, next_pos, wall_mask)
        
        # Update physical position
        self.x, self.y = next_pos

        # Update previous cell's walls too (Bidirectional logic)
        # In reality, you'd do this when LEAVING the previous cell.
        prev_mask = self.mem[cy][cx][L_WALL]
        if dx == 1: prev_mask &= ~WALL_E
        elif dx == -1: prev_mask &= ~WALL_W
        elif dy == 1: prev_mask &= ~WALL_S
        elif dy == -1: prev_mask &= ~WALL_N
        self.mem[cy][cx][L_WALL] = prev_mask

    def solve_grid_astar(self, start, goal):
        pq = [(0, start, (0,0))]
        g_scores = {}
        g_scores[(start, (0,0))] = 0
        came_from = {} 
        final_state = None

        while pq:
            cost, curr, arr_dir = heapq.heappop(pq)
            if curr == goal:
                final_state = (curr, arr_dir); break

            cx, cy = curr
            walls = self.mem[cy][cx][L_WALL]
            
            for i, (dx, dy) in enumerate([(1,0), (0,1), (-1,0), (0,-1)]):
                mask = [WALL_E, WALL_S, WALL_W, WALL_N][i]
                if not (walls & mask):
                    nx, ny = cx + dx, cy + dy
                    new_cost = cost + STEP_COST
                    if arr_dir != (0,0) and (dx, dy) != arr_dir:
                        new_cost += TURN_PENALTY
                    
                    state = ((nx, ny), (dx, dy))
                    if new_cost < g_scores.get(state, float('inf')):
                        g_scores[state] = new_cost
                        came_from[state] = (curr, arr_dir)
                        h = (abs(nx - goal[0]) + abs(ny - goal[1])) * STEP_COST
                        heapq.heappush(pq, (new_cost + h, (nx, ny), (dx, dy)))

        if final_state:
            self.final_path = []
            curr_key = final_state
            while curr_key in came_from:
                self.final_path.append(curr_key[0])
                curr_key = came_from[curr_key]
            self.final_path.append(start)
            self.final_path.reverse()
            self.print_matrix()

    def print_matrix(self):
        print("\n// ---- ARDUINO MATRIX ----")
        print(f"int maze[{MAZE_SIZE}][{MAZE_SIZE}][3] = {{")
        for y in range(MAZE_SIZE):
            print("  {", end="")
            for x in range(MAZE_SIZE):
                d = self.mem[y][x][L_DIST]
                w = self.mem[y][x][L_WALL]
                print(f"{{{d},0,{w}}}", end="")
                if x < MAZE_SIZE-1: print(", ", end="")
            print("},")
        print("};")

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Modular Explorer")
    clock = pygame.time.Clock()
    
    real_maze = get_maze_layout()
    robot = Robot((1,1))

    while True:
        clock.tick(FPS)
        for e in pygame.event.get():
            if e.type == pygame.QUIT: pygame.quit(); return

        robot.move_hardware_simulation(real_maze)

        screen.fill(BG_COLOR)
        
        # Draw Visited
        for (x,y) in robot.visited:
            pygame.draw.rect(screen, VISITED_COLOR, (x*CELL+2, y*CELL+2, CELL-4, CELL-4))

        # Draw Walls from Memory
        for y in range(MAZE_SIZE):
            for x in range(MAZE_SIZE):
                w = robot.mem[y][x][L_WALL]
                if w == 15: continue
                px, py = x*CELL, y*CELL
                if w & WALL_N: pygame.draw.line(screen, WALL_COLOR, (px,py), (px+CELL,py), 2)
                if w & WALL_E: pygame.draw.line(screen, WALL_COLOR, (px+CELL,py), (px+CELL,py+CELL), 2)
                if w & WALL_S: pygame.draw.line(screen, WALL_COLOR, (px,py+CELL), (px+CELL,py+CELL), 2)
                if w & WALL_W: pygame.draw.line(screen, WALL_COLOR, (px,py), (px,py+CELL), 2)

        # Draw Path
        if robot.final_path:
            pts = [(x*CELL+CELL//2, y*CELL+CELL//2) for (x,y) in robot.final_path]
            pygame.draw.lines(screen, PATH_COLOR, False, pts, 4)
            pygame.draw.lines(screen, (255,255,255), False, pts, 1)

        # Draw Robot
        pygame.draw.circle(screen, ROBOT_COLOR, (robot.x*CELL+CELL//2, robot.y*CELL+CELL//2), 10)
        pygame.display.flip()

if __name__ == "__main__":
    main()