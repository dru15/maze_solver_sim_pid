import pygame
import numpy as np
import heapq

# ---------------- SETTINGS ----------------
CELL = 60
MAZE_SIZE = 10
WIDTH, HEIGHT = MAZE_SIZE * CELL, MAZE_SIZE * CELL
FPS = 40

# encoder & costs
STEP_TICKS = 1000
TURN_PENALTY = 5000

# directions: 0=E,1=S,2=W,3=N
DIRS = [(1,0),(0,1),(-1,0),(0,-1)]
DIR_NAMES = ["E", "S", "W", "N"]

# turn mask bits: N=1, E=2, S=4, W=8
BIT_N, BIT_E, BIT_S, BIT_W = 1,2,4,8

# colors
BG = (12,12,20)
GRID = (30,35,50)
VISITED = (24,28,40)
NODE_COLOR = (255,200,50)
EDGE_COLOR = (60,200,120)
PATH_COLOR = (255,0,200)
ROBOT_COLOR = (255,255,80)
TEXT_COLOR = (200,200,220)

# ---------------- MAZE ----------------
def get_maze_layout():
    return np.array([
        [1,1,1,1,1,1,1,1,1,1],
        [1,0,0,0,0,0,1,0,0,1],
        [1,0,1,1,1,0,1,0,1,1],
        [1,0,0,0,1,0,0,0,0,1],
        [1,1,1,0,1,1,1,1,0,1],
        [1,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,0,1,0,1],
        [1,0,0,0,0,0,0,1,0,1],
        [1,0,1,1,1,1,0,0,0,1],
        [1,1,1,1,1,1,1,1,1,1]
    ])

# ---------------- Robot / Explorer ----------------
class LFRExplorer:
    def __init__(self, start=(1,1)):
        self.x, self.y = start
        self.mem_walls = np.full((MAZE_SIZE, MAZE_SIZE), 15, dtype=int)
        self.visited_cells = set()
        self.visited_cells.add((self.x, self.y))

        # Node Graph Data
        self.cell_to_node = {}      # (x,y) -> node_id
        self.node_to_cell = []      # node_id -> (x,y)
        self.edges = {}             # edges[node_A][node_B] = distance

        self.current_node = None    
        self.last_node = None       
        self.dist_acc = 0           

        self.walking = False
        self.walk_dir = None        
        self.walk_from_node = None  

        self.stack = []
        self.finished = False

    def is_node_cell(self, cell, maze):
        x,y = cell
        deg = 0
        neighbors = []
        for d,(dx,dy) in enumerate(DIRS):
            nx,ny = x+dx, y+dy
            if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE and maze[ny][nx] == 0:
                deg += 1
                neighbors.append(d)
        if deg != 2: return True
        if len(neighbors) == 2 and (neighbors[0] + 2) % 4 != neighbors[1]:
            return True
        return False

    def ensure_node(self, cell):
        if cell in self.cell_to_node:
            return self.cell_to_node[cell]
        nid = len(self.node_to_cell)
        self.cell_to_node[cell] = nid
        self.node_to_cell.append(cell)
        self.edges[nid] = {}
        return nid

    def start_exploration(self, maze):
        start_cell = (self.x, self.y)
        nid = self.ensure_node(start_cell)
        self.current_node = nid
        self.last_node = nid
        self.dist_acc = 0

        # Push valid exits
        x,y = start_cell
        for d,(dx,dy) in enumerate(DIRS):
            nx,ny = x+dx, y+dy
            if 0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE and maze[ny][nx] == 0:
                self.stack.append((nid, d))

    def start_walk(self, dir_to_walk):
        self.walking = True
        self.walk_dir = dir_to_walk
        self.walk_from_node = self.current_node

    def step_walk_one_cell(self, maze):
        dx,dy = DIRS[self.walk_dir]
        nx,ny = self.x + dx, self.y + dy

        if not (0 <= nx < MAZE_SIZE and 0 <= ny < MAZE_SIZE and maze[ny][nx] == 0):
            self.walking = False
            return

        self.x, self.y = nx, ny
        self.visited_cells.add((self.x, self.y))
        self.dist_acc += STEP_TICKS

        if self.is_node_cell((self.x, self.y), maze):
            nid = self.ensure_node((self.x, self.y))
            
            # Record Edge
            a, b = self.walk_from_node, nid
            ticks = self.dist_acc if self.dist_acc>0 else STEP_TICKS
            self.edges.setdefault(a, {})[b] = ticks
            self.edges.setdefault(b, {})[a] = ticks

            self.last_node = a
            self.current_node = b
            self.dist_acc = 0
            self.walking = False

            # Add new tasks
            x0,y0 = self.node_to_cell[b]
            for d,(dx,dy) in enumerate(DIRS):
                nx2, ny2 = x0+dx, y0+dy
                if 0 <= nx2 < MAZE_SIZE and 0 <= ny2 < MAZE_SIZE and maze[ny2][nx2] == 0:
                    # Don't go back immediately
                    back_dx, back_dy = DIRS[(self.walk_dir+2)%4]
                    if (nx2,ny2) == (x0+back_dx, y0+back_dy): continue
                    self.stack.append((b,d))

    def exploration_step(self, maze):
        if self.current_node is None:
            self.start_exploration(maze)
            return

        if self.walking:
            self.step_walk_one_cell(maze)
            return

        if not self.stack:
            self.finished = True
            return

        node_id, dir_to_explore = self.stack.pop()
        
        if node_id != self.current_node:
            tx,ty = self.node_to_cell[node_id]
            self.x, self.y = tx, ty
            self.current_node = node_id
        
        self.start_walk(dir_to_explore)

    # ---------------- 3D ADJACENCY MATRIX GENERATOR ----------------
    def print_adjacency_matrix(self):
        n = len(self.node_to_cell)
        print(f"\n// --- 3D ADJACENCY MATRIX ({n} Nodes) ---")
        print("// Format: {Distance, Dir_Index, Turn_Mask}")
        print(f"// Nodes: {n}, Directions: 4, Data: 3")
        print(f"int graph[{n}][4][3] = {{")
        
        for i in range(n):
            cx, cy = self.node_to_cell[i]
            
            # 1. Calculate Turn Mask for this Node
            # (Check all connected neighbors)
            mask = 0
            neighbors = self.edges.get(i, {})
            
            # Also check which directions have edges
            # We map neighbors to directions E, S, W, N
            dir_map = {} # Dir_Index -> Neighbor_ID
            
            for nb_id in neighbors:
                nx, ny = self.node_to_cell[nb_id]
                dx, dy = nx - cx, ny - cy
                
                # Determine direction of neighbor
                d_idx = -1
                if dx > 0: d_idx = 0   # East
                elif dy > 0: d_idx = 1 # South
                elif dx < 0: d_idx = 2 # West
                elif dy < 0: d_idx = 3 # North
                
                if d_idx != -1:
                    dir_map[d_idx] = nb_id
                    # Update mask
                    if d_idx == 0: mask |= BIT_E
                    elif d_idx == 1: mask |= BIT_S
                    elif d_idx == 2: mask |= BIT_W
                    elif d_idx == 3: mask |= BIT_N

            # 2. Print Rows for E, S, W, N
            print(f"  // Node {i} {self.node_to_cell[i]}")
            print("  {")
            for d in range(4):
                if d in dir_map:
                    nb = dir_map[d]
                    dist = neighbors[nb]
                    print(f"    {{{dist}, {d}, {mask}}}", end="")
                else:
                    print(f"    {{0, 0, 0}}", end="")
                
                if d < 3: print(",")
                else: print("")
            
            print("  }", end="")
            if i < n-1: print(",")
            print("")
            
        print("};")

# ---------------- PYGAME VISUAL ----------------
def draw_grid(screen):
    for i in range(MAZE_SIZE+1):
        pygame.draw.line(screen, GRID, (0,i*CELL), (WIDTH,i*CELL))
        pygame.draw.line(screen, GRID, (i*CELL,0), (i*CELL,HEIGHT))

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("LFR Graph Generator")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Consolas", 16)

    maze = get_maze_layout()
    bot = LFRExplorer((1,1))

    running = True
    printed = False

    while running:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT: running = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE: bot.finished = True
                if ev.key == pygame.K_p:
                    bot.print_adjacency_matrix()
                    printed = True

        if not bot.finished:
            bot.exploration_step(maze)
        else:
            if not printed:
                bot.print_adjacency_matrix()
                printed = True

        screen.fill(BG)
        draw_grid(screen)

        # Draw Edges
        for a, nbrs in bot.edges.items():
            ax,ay = bot.node_to_cell[a]
            for b in nbrs:
                bx,by = bot.node_to_cell[b]
                pygame.draw.line(screen, EDGE_COLOR, (ax*CELL+30,ay*CELL+30), (bx*CELL+30,by*CELL+30), 4)

        # Draw Nodes
        for nid, (nx,ny) in enumerate(bot.node_to_cell):
            cx, cy = nx*CELL + 30, ny*CELL + 30
            pygame.draw.circle(screen, NODE_COLOR, (cx,cy), 12)
            screen.blit(font.render(str(nid), True, TEXT_COLOR), (cx-8, cy-8))

        # Robot
        pygame.draw.circle(screen, ROBOT_COLOR, (bot.x*CELL+30, bot.y*CELL+30), 9)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()