import heapq
import numpy as np
# from scipy.ndimage import binary_dilation


def pathfind(scanmapper, start_xu, start_yu, goal_xu, goal_yu, buffer=10, blockymode=True, domain=10): #buffer=7, 5
    '''
    Returns the path to follow as a list of yx coords
    - pass in scanmapper.scanmap
    '''

    pathfindgrid = scanmapper.scanmap.copy()

    #inflate AROUND DRONEPOS (domain expansion)
    rows, cols = pathfindgrid.shape
    y, x = np.ogrid[:rows, :cols]
    distance_mask = (y - start_yu)**2 + (x - start_xu)**2 <= domain**2
    pathfindgrid[distance_mask] = 1
    # scanmapper.scanmap[distance_mask] = 4


    #inflate OBSTACLES ONLY in pathfindgrid (scanmap)
    # y, x = np.ogrid[-buffer:buffer+1, -buffer:buffer+1]
    # structure = x*x + y*y <= buffer*buffer
    # pathfindgrid[binary_dilation(pathfindgrid == 2,structure=structure)] = 2

    pathfindgrid[pathfindgrid==0] = 2
    pathfindgrid[pathfindgrid>2] = 1
    pathfindgrid[pathfindgrid==1] = 0
    pathfindgrid[pathfindgrid==2] = 1   

    if pathfindgrid[start_yu, start_xu] == 1:
        print("start blocked")
        return None
    if pathfindgrid[goal_yu, goal_xu] == 1:
        print("goal blocked")
        return None

    return astar(
        pathfindgrid,
        (start_xu, start_yu),
        (goal_xu, goal_yu),
        blockymode
    )

def simplifypath(path): #path is in yx

    if len(path) < 3: return path

    simplified = [path[0]]

    # Initial direction
    prev_dy = path[1][0] - path[0][0] #col
    prev_dx = path[1][1] - path[0][1] #row

    for i in range(1, len(path) - 1):
        curr_dy = path[i + 1][0] - path[i][0]
        curr_dx = path[i + 1][1] - path[i][1]

        # Direction changed → corner
        if (curr_dx, curr_dy) != (prev_dx, prev_dy): simplified.append(path[i])

        prev_dx, prev_dy = curr_dx, curr_dy

    simplified.append(path[-1])

    return simplified



def astar(
    grid: np.ndarray,
    start: tuple,
    goal: tuple,
    blockymode=True
):
    if blockymode: return _astar_directional(grid, start, goal)
    else: return _astar_simple(grid, start, goal)


def _astar_simple(grid, start, goal):
    """Non-directional A* — returns shortest path by cell count."""

    rows, cols = grid.shape
    start = (start[1], start[0])
    goal  = (goal[1],  goal[0])

    def h(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

    neighbours = [(-1,0),(1,0),(0,-1),(0,1)]

    open_set = []
    counter  = 0
    heapq.heappush(open_set, (h(start, goal), counter, start))

    came_from = {}
    g_score   = {start: 0.0}
    closed_set = set()

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue
        closed_set.add(current)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        curr_row, curr_col = current

        for dr, dc in neighbours:
            nb_row, nb_col = curr_row + dr, curr_col + dc

            if not (0 <= nb_row < rows and 0 <= nb_col < cols):
                continue
            if grid[nb_row, nb_col] == 1:
                continue

            nb_state      = (nb_row, nb_col)
            tentative_g   = g_score[current] + 1.0

            if tentative_g < g_score.get(nb_state, float("inf")):
                came_from[nb_state] = current
                g_score[nb_state]   = tentative_g
                counter += 1
                heapq.heappush(open_set,
                    (tentative_g + h(nb_state, goal), counter, nb_state))

    return None


def _astar_directional(grid, start, goal, turn_penalty=0.5, uturn_penalty=1.0, tiebreaker=1.001):
    """
    Returns the path found as a list of tuple coords
    - give start and goal coords in scanmapXY coords
    """

    rows, cols = grid.shape

    start = (start[1], start[0])
    goal = (goal[1], goal[0])

    # if not (0 <= start[0] < rows and 0 <= start[1] < cols):
    #     print("Start out of bounds")
    #     return None

    # if not (0 <= goal[0] < rows and 0 <= goal[1] < cols):
    #     print("Goal out of bounds")
    #     return None

    # if grid[start[0], start[1]] == 1:
    #     print("Start blocked")
    #     return None

    # if grid[goal[0], goal[1]] == 1:
    #     print("Goal blocked")
    #     return None

    # ----------------------------
    # Manhattan heuristic
    # ----------------------------

    def h(a, b): return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ----------------------------
    # Directions
    # ----------------------------

    # 0=N, 1=S, 2=W, 3=E
    neighbours = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    opposite = {
        0: 1,
        1: 0,
        2: 3,
        3: 2,
    }

    # ----------------------------
    # Initial state
    # ----------------------------

    start_state = (start[0], start[1], None)
    open_set = []
    counter = 0
    start_f = h(start, goal)

    heapq.heappush(
        open_set,
        (start_f, counter, start_state)
    )

    came_from = {}
    g_score = {
        start_state: 0.0
    }

    closed_set = set()

    # ----------------------------
    # Main search loop
    # ----------------------------

    while open_set:

        current_f, _, current = heapq.heappop(open_set)

        # Skip stale heap entries
        if current in closed_set:
            continue

        closed_set.add(current)

        curr_row, curr_col, curr_dir = current

        # ------------------------
        # Goal reached
        # ------------------------

        if (curr_row, curr_col) == goal:

            path = []

            while current in came_from:
                path.append((current[0], current[1]))
                current = came_from[current]

            path.append(start)
            path.reverse()
            # return [(x, y) for y, x in path]
            return path

        # ------------------------
        # Expand neighbours
        # ------------------------

        for dir_idx, (dr, dc) in enumerate(neighbours):

            nb_row = curr_row + dr
            nb_col = curr_col + dc

            # Bounds check
            if not (
                0 <= nb_row < rows and
                0 <= nb_col < cols
            ):
                continue

            # Obstacle check
            if grid[nb_row, nb_col] == 1:
                continue

            # --------------------
            # Direction penalties
            # --------------------

            turned = (
                curr_dir is not None and
                dir_idx != curr_dir
            )

            is_u_turn = (
                curr_dir is not None and
                dir_idx == opposite[curr_dir]
            )

            penalty = 0.0

            if is_u_turn:
                penalty = uturn_penalty

            elif turned:
                penalty = turn_penalty

            move_cost = 1.0 + penalty

            tentative_g = g_score[current] + move_cost

            nb_state = (
                nb_row,
                nb_col,
                dir_idx
            )

            # --------------------
            # Better path found
            # --------------------

            if tentative_g < g_score.get(nb_state, float("inf")):

                came_from[nb_state] = current

                g_score[nb_state] = tentative_g

                heuristic = h(
                    (nb_row, nb_col),
                    goal
                )

                # Slight tie-breaking bias
                f = tentative_g + (heuristic * tiebreaker)

                counter += 1

                heapq.heappush(
                    open_set,
                    (f, counter, nb_state)
                )

    # No path found
    return None