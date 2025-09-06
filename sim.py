import pygame
import numpy as np
import math
import random
import heapq
import time

CELL = 30
MAZE_W, MAZE_H = 15, 15
WIDTH, HEIGHT = MAZE_W * CELL, MAZE_H * CELL
FPS = 60

ROBOT_RADIUS = 0.28
MAX_SPEED = 5.0
MAX_ANGULAR = 3.2

SENSOR_RANGE = 4.0
SENSOR_STEP = 0.02

HEADING_KP = 10.0
HEADING_KI = 0.0
HEADING_KD = 0.1

CENTER_KP = 5.0
CENTER_KI = 0.0
CENTER_KD = 0.5

WHITE = (255,255,255)
BLACK = (0,0,0)
DARKGRAY = (50,50,50)
GRAY = (180,180,180)
BLUE = (40,120,255)
GREEN = (0,200,0)
RED = (200,0,0)
ROBOT_COLOR = (10,10,180)
HEADING_COLOR = (255,255,255)

def angle_wrap(a):
    return (a + math.pi) % (2*math.pi) - math.pi

def generate_maze(w, h):
    maze = np.ones((h, w), dtype=int)
    def carve(cx, cy, visited):
        visited.add((cx, cy))
        maze[cy, cx] = 0
        dirs = [(2,0),(-2,0),(0,2),(0,-2)]
        random.shuffle(dirs)
        for dx,dy in dirs:
            nx, ny = cx + dx, cy + dy
            if 1 <= nx < w-1 and 1 <= ny < h-1 and (nx,ny) not in visited:
                maze[cy + dy//2, cx + dx//2] = 0
                carve(nx, ny, visited)
    carve(1,1,set())
    return maze

def get_distance_to_wall(maze, x, y, angle, max_range=SENSOR_RANGE):
    dist = 0.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    while dist < max_range:
        tx = x + cos_a * dist
        ty = y + sin_a * dist
        if tx < 0 or ty < 0 or tx >= MAZE_W or ty >= MAZE_H:
            return dist
        gx, gy = int(tx), int(ty)
        if maze[gy, gx] == 1:
            return dist
        dist += SENSOR_STEP
    return max_range

def astar_on_seen(seen_map, start_cell, goal_cell):
    h = lambda a,b: abs(a[0]-b[0]) + abs(a[1]-b[1])
    open_heap = []
    heapq.heappush(open_heap, (h(start_cell, goal_cell), start_cell))
    came = {}
    gscore = {start_cell:0}
    visited = set()
    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal_cell:
            path = []
            while current in came:
                path.append(current)
                current = came[current]
            path.append(start_cell)
            path.reverse()
            return path
        if current in visited:
            continue
        visited.add(current)
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            n = (current[0]+dx, current[1]+dy)
            if not (0 <= n[0] < MAZE_W and 0 <= n[1] < MAZE_H): 
                continue
            if seen_map[n[1], n[0]] != 0:
                continue
            tentative = gscore[current] + 1
            if tentative < gscore.get(n, 1e9):
                came[n] = current
                gscore[n] = tentative
                heapq.heappush(open_heap, (tentative + h(n, goal_cell), n))
    return []

class PID:
    def __init__(self, kp, ki=0.0, kd=0.0, out_min=None, out_max=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = 0.0
        self.last = 0.0
        self.out_min, self.out_max = out_min, out_max

    def update(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.last) / dt if dt>0 else 0.0
        out = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.last = error
        if self.out_min is not None: out = max(self.out_min, out)
        if self.out_max is not None: out = min(self.out_max, out)
        return out

def circle_collides(maze, cx, cy, r):
    ix, iy = int(cx), int(cy)
    for gy in range(iy-1, iy+2):
        for gx in range(ix-1, ix+2):
            if gx < 0 or gy < 0 or gx >= MAZE_W or gy >= MAZE_H:
                continue
            if maze[gy, gx] == 1:
                closest_x = max(gx, min(cx, gx+1))
                closest_y = max(gy, min(cy, gy+1))
                dx = cx - closest_x
                dy = cy - closest_y
                if dx*dx + dy*dy < r*r:
                    return True
    return False

class Robot:
    def __init__(self, x_cell, y_cell):
        self.x = x_cell + 0.5
        self.y = y_cell + 0.5
        self.theta = 0.0
        self.radius = ROBOT_RADIUS
        self.explored = set()
        self.explored.add((int(self.x), int(self.y)))
        self.heading_pid = PID(HEADING_KP, HEADING_KI, HEADING_KD, out_min=-MAX_ANGULAR, out_max=MAX_ANGULAR)
        self.center_pid = PID(CENTER_KP, CENTER_KI, CENTER_KD, out_min=-1.5, out_max=1.5)
        self.mode = 'explore'
        self.left_hand_dir = 0
        self.target_cell = (int(self.x), int(self.y))
        self.astar_path = []
        self.astar_index = 0

    def set_target_cell(self, cell):
        self.target_cell = cell

    def at_cell_center(self):
        tx = self.target_cell[0] + 0.5
        ty = self.target_cell[1] + 0.5
        return math.hypot(self.x - tx, self.y - ty) < 0.12

    def sense(self, maze):
        left = get_distance_to_wall(maze, self.x, self.y, self.theta + math.pi/2)
        front = get_distance_to_wall(maze, self.x, self.y, self.theta)
        right = get_distance_to_wall(maze, self.x, self.y, self.theta - math.pi/2)
        return left, front, right

    def step(self, maze, dt):
        lx, fx, rx = self.sense(maze)

        # Desired heading
        tx = self.target_cell[0] + 0.5
        ty = self.target_cell[1] + 0.5
        desired_heading = math.atan2(ty - self.y, tx - self.x)
        heading_err = angle_wrap(desired_heading - self.theta)
        heading_cmd = self.heading_pid.update(heading_err, dt)

        # Centering correction
        lateral_error = lx - rx
        center_cmd = self.center_pid.update(lateral_error, dt)

        # Blend heading + centering
        ang_vel = heading_cmd + center_cmd * 0.4
        ang_vel = max(-MAX_ANGULAR, min(MAX_ANGULAR, ang_vel))

        # Forward speed depends on turn + clearance
        turn_factor = max(0.35, 1.0 - min(abs(ang_vel) / MAX_ANGULAR, 0.95))
        front_factor = max(0.2, min(fx / 2.0, 1.0))
        forward = MAX_SPEED * turn_factor * front_factor

        self._integrate_motion(forward, ang_vel, maze, dt)
        self.explored.add((int(self.x), int(self.y)))

    def _integrate_motion(self, forward, ang_vel, maze, dt):
        new_theta = self.theta + ang_vel * dt
        nx = self.x + math.cos(new_theta) * forward * dt
        ny = self.y + math.sin(new_theta) * forward * dt
        if not circle_collides(maze, nx, ny, self.radius):
            self.x, self.y, self.theta = nx, ny, new_theta
        else:
            self.theta = new_theta

    def explore_decision(self, maze):
        dirs = [(1,0),(0,1),(-1,0),(0,-1)]
        x, y = int(self.x), int(self.y)
        d = self.left_hand_dir
        for turn in [-1, 0, 1, 2]:
            nd = (d + turn) % 4
            dx, dy = dirs[nd]
            nx, ny = x + dx, y + dy
            if 0 <= nx < MAZE_W and 0 <= ny < MAZE_H and maze[ny, nx] != 1:
                self.left_hand_dir = nd
                self.set_target_cell((nx, ny))
                return

    def start_astar_follow(self, seen_map, start_cell, goal_cell):
        path = astar_on_seen(seen_map, start_cell, goal_cell)
        if path:
            self.astar_path = path
            self.astar_index = 0
            self.mode = 'astar'
            self.set_target_cell(self.astar_path[0])

    def follow_astar_step(self):
        if not self.astar_path: return
        if self.astar_index < len(self.astar_path):
            self.set_target_cell(self.astar_path[self.astar_index])
            if self.at_cell_center():
                self.astar_index += 1

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Continuous PID Explorer + A*")
    clock = pygame.time.Clock()

    maze = generate_maze(MAZE_W, MAZE_H)
    start_cell = (1, 1)
    goal_cell = (MAZE_W-2, MAZE_H-2)
    maze[goal_cell[1], goal_cell[0]] = 0

    robot = Robot(start_cell[0], start_cell[1])
    robot.left_hand_dir = 0
    robot.set_target_cell(start_cell)

    running = True
    last_time = time.time()

    while running:
        now = time.time()
        dt = now - last_time
        last_time = now
        if dt <= 0: dt = 1.0/FPS

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        if robot.mode == 'explore':
            if robot.at_cell_center():
                heading_dir = min(range(4), key=lambda d: abs(angle_wrap(robot.theta - (d * math.pi/2))))
                robot.left_hand_dir = heading_dir
                robot.explore_decision(maze)

            robot.step(maze, dt)

            if (int(robot.x), int(robot.y)) == goal_cell:
                seen_map = np.ones_like(maze)
                for (sx, sy) in robot.explored:
                    seen_map[sy, sx] = 0
                seen_map[goal_cell[1], goal_cell[0]] = 0
                astar_path = astar_on_seen(seen_map, start_cell, goal_cell)
                if astar_path:
                    robot.x = start_cell[0] + 0.5
                    robot.y = start_cell[1] + 0.5
                    robot.theta = 0.0
                    robot.astar_path = astar_path
                    robot.astar_index = 0
                    robot.mode = 'astar'
                    robot.set_target_cell(robot.astar_path[0])
                else:
                    robot.mode = 'idle'

        elif robot.mode == 'astar':
            if robot.at_cell_center():
                if robot.astar_index < len(robot.astar_path):
                    robot.set_target_cell(robot.astar_path[robot.astar_index])
                    robot.astar_index += 1
            robot.step(maze, dt)
            if (int(robot.x), int(robot.y)) == goal_cell:
                robot.mode = 'idle'

        # Drawing
        screen.fill(BLACK)
        for y in range(MAZE_H):
            for x in range(MAZE_W):
                rect = pygame.Rect(x*CELL, y*CELL, CELL, CELL)
                if maze[y,x] == 1:
                    pygame.draw.rect(screen, BLACK, rect)
                else:
                    pygame.draw.rect(screen, DARKGRAY, rect)

        for (sx, sy) in robot.explored:
            pygame.draw.rect(screen, GRAY, (sx*CELL, sy*CELL, CELL, CELL))

        if robot.astar_path:
            for (px, py) in robot.astar_path:
                pygame.draw.rect(screen, BLUE, (px*CELL, py*CELL, CELL, CELL))

        pygame.draw.rect(screen, GREEN, (start_cell[0]*CELL, start_cell[1]*CELL, CELL, CELL))
        pygame.draw.rect(screen, RED, (goal_cell[0]*CELL, goal_cell[1]*CELL, CELL, CELL))

        rx = int(robot.x * CELL)
        ry = int(robot.y * CELL)
        pygame.draw.circle(screen, ROBOT_COLOR, (rx, ry), max(3, int(robot.radius * CELL)))
        hx = int((robot.x + math.cos(robot.theta)*0.5)*CELL)
        hy = int((robot.y + math.sin(robot.theta)*0.5)*CELL)
        pygame.draw.line(screen, HEADING_COLOR, (rx, ry), (hx, hy), 2)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()