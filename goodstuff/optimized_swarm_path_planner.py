import math
import heapq
import random

import numpy as np

try:
    from Astar import pathfind, simplifypath
except ModuleNotFoundError:
    def _binary_dilation(mask, radius_cells):
        mask = np.asarray(mask, dtype=bool)
        out = mask.copy()

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx) + (dy * dy) > radius_cells * radius_cells:
                    continue

                src_y0 = max(0, -dy)
                src_y1 = min(mask.shape[0], mask.shape[0] - dy)
                src_x0 = max(0, -dx)
                src_x1 = min(mask.shape[1], mask.shape[1] - dx)

                dst_y0 = src_y0 + dy
                dst_y1 = src_y1 + dy
                dst_x0 = src_x0 + dx
                dst_x1 = src_x1 + dx

                out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]

        return out

    def _astar_grid(grid, start_xy, goal_xy):
        start = (start_xy[1], start_xy[0])
        goal = (goal_xy[1], goal_xy[0])

        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dy, dx in neighbours:
                ny = current[0] + dy
                nx = current[1] + dx

                if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                    continue
                if grid[ny, nx] == 1:
                    continue

                next_cell = (ny, nx)
                tentative_g = g_score[current] + 1
                if tentative_g < g_score.get(next_cell, float("inf")):
                    came_from[next_cell] = current
                    g_score[next_cell] = tentative_g
                    priority = tentative_g + heuristic(next_cell, goal)
                    heapq.heappush(open_set, (priority, next_cell))

        return None

    def pathfind(obstaclemap, start_xu, start_yu, goal_xu, goal_yu, buffer=10, blockypath=True):
        grid = obstaclemap.copy()
        grid[_binary_dilation(grid == 1, buffer)] = 1

        if grid[start_yu, start_xu] == 1 or grid[goal_yu, goal_xu] == 1:
            return None

        return _astar_grid(grid, (start_xu, start_yu), (goal_xu, goal_yu))

    def simplifypath(path):
        if path is None or len(path) < 3:
            return path

        simplified = [path[0]]
        prev_dy = path[1][0] - path[0][0]
        prev_dx = path[1][1] - path[0][1]

        for i in range(1, len(path) - 1):
            curr_dy = path[i + 1][0] - path[i][0]
            curr_dx = path[i + 1][1] - path[i][1]
            if (curr_dy, curr_dx) != (prev_dy, prev_dx):
                simplified.append(path[i])
            prev_dy, prev_dx = curr_dy, curr_dx

        simplified.append(path[-1])
        return simplified


FREE = 1
OBSTACLE = 2
SEARCHED = 8
DEFAULT_CLEARANCE_CELLS = 5
DEFAULT_BUFFER_CELLS = 5
MIN_ROUTE_BUFFER_CELLS = 3


def dilate_mask(mask, radius_cells):
    """Dilate a boolean mask by a circular radius in grid cells."""
    mask = np.asarray(mask, dtype=bool)
    out = mask.copy()

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if (dx * dx) + (dy * dy) > radius_cells * radius_cells:
                continue

            src_y0 = max(0, -dy)
            src_y1 = min(mask.shape[0], mask.shape[0] - dy)
            src_x0 = max(0, -dx)
            src_x1 = min(mask.shape[1], mask.shape[1] - dx)

            dst_y0 = src_y0 + dy
            dst_y1 = src_y1 + dy
            dst_x0 = src_x0 + dx
            dst_x1 = src_x1 + dx

            out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]

    return out


def astar_grid(grid, start_xy, goal_xy):
    """Small local A* over a binary grid where 1 means blocked."""
    start = (start_xy[1], start_xy[0])
    goal = (goal_xy[1], goal_xy[0])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = current[0] + dy
            nx = current[1] + dx

            if not (0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]):
                continue
            if grid[ny, nx] == 1:
                continue

            next_cell = (ny, nx)
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(next_cell, float("inf")):
                came_from[next_cell] = current
                g_score[next_cell] = tentative_g
                priority = tentative_g + heuristic(next_cell, goal)
                heapq.heappush(open_set, (priority, next_cell))

    return None


def scanmap_to_obstaclemap(scanmap):
    """Convert ScanMapper values into Astar values: 0=free, 1=blocked."""
    return (scanmap == OBSTACLE).astype(np.uint8)


def world_xy_to_grid_xy(scanmapper, x_east, y_north):
    """Convert world (x=east, y=north) metres to grid (x, y) cells."""
    return scanmapper.worldNE_to_scanmapXY(y_north, x_east)


def grid_yx_to_world_xy(scanmapper, y_cell, x_cell):
    """Convert Astar (y, x) cell to world (x=east, y=north) metres."""
    north, east = scanmapper.scanmapXY_to_worldNE(x_cell, y_cell)
    return east, north


def path_cells_to_world_xy(scanmapper, path_yx):
    """Convert Astar path cells into waypoint tuples for movement code."""
    return [grid_yx_to_world_xy(scanmapper, y, x) for y, x in path_yx]


def plan_path_to_grid_goal(
    scanmapper,
    start_xy,
    goal_xy_cell,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    extra_blocked_mask=None,
):
    """Plan from world (x, y) start to grid (x, y) goal."""
    start_x, start_y = world_xy_to_grid_xy(
        scanmapper,
        x_east=start_xy[0],
        y_north=start_xy[1],
    )
    goal_x, goal_y = goal_xy_cell

    if extra_blocked_mask is not None:
        real_obstacles = scanmapper.scanmap == OBSTACLE
        blocked = dilate_mask(real_obstacles, buffer_cells)
        blocked |= np.asarray(extra_blocked_mask, dtype=bool)
        blocked[start_y, start_x] = False
        blocked[goal_y, goal_x] = False

        if blocked[start_y, start_x] or blocked[goal_y, goal_x]:
            return None

        path_yx = astar_grid(
            blocked.astype(np.uint8),
            (start_x, start_y),
            (goal_x, goal_y),
        )
    else:
        obstaclemap = scanmap_to_obstaclemap(scanmapper.scanmap)
        path_yx = pathfind(
            obstaclemap=obstaclemap,
            start_xu=start_x,
            start_yu=start_y,
            goal_xu=goal_x,
            goal_yu=goal_y,
            buffer=buffer_cells,
            blockypath=True,
        )

    if path_yx is None:
        return None

    return path_cells_to_world_xy(scanmapper, simplifypath(path_yx))


def route_length(path_xy):
    """Approximate path length in metres."""
    if not path_xy or len(path_xy) < 2:
        return 0.0

    total = 0.0
    for prev, curr in zip(path_xy, path_xy[1:]):
        total += math.hypot(curr[0] - prev[0], curr[1] - prev[1])
    return total


def route_center_cells(scanmapper, route_xy):
    """Trace waypoint segments into grid cells along the route centerline."""
    cells = []
    if not route_xy:
        return cells

    for start_xy, end_xy in zip(route_xy, route_xy[1:]):
        distance = math.hypot(end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
        steps = max(1, int(math.ceil(distance / max(scanmapper.resolution, 1e-6))))

        for step in range(steps + 1):
            t = step / steps
            x = start_xy[0] + ((end_xy[0] - start_xy[0]) * t)
            y = start_xy[1] + ((end_xy[1] - start_xy[1]) * t)
            grid_x, grid_y = world_xy_to_grid_xy(scanmapper, x, y)
            cell = (grid_y, grid_x)
            if not cells or cells[-1] != cell:
                cells.append(cell)

    if len(route_xy) == 1:
        grid_x, grid_y = world_xy_to_grid_xy(scanmapper, route_xy[0][0], route_xy[0][1])
        cells.append((grid_y, grid_x))

    return cells


def inflate_grid_cells(scanmapper, cells_yx, radius_m):
    """Inflate grid cells by radius_m and return a boolean mask."""
    mask = np.zeros(scanmapper.scanmap.shape, dtype=bool)
    radius_cells = max(0, int(round(radius_m / scanmapper.resolution)))

    for center_y, center_x in cells_yx:
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if (dx * dx) + (dy * dy) > radius_cells * radius_cells:
                    continue

                y = center_y + dy
                x = center_x + dx
                if 0 <= y < scanmapper.height and 0 <= x < scanmapper.width:
                    mask[y, x] = True

    return mask


def trace_route_scan_mask(scanmapper, route_xy, scan_radius_m=None):
    """Return cells scanned by flying along route_xy."""
    scan_radius = get_search_radius_m(scanmapper, scan_radius_m)
    route_cells = route_center_cells(scanmapper, route_xy)
    scanned_mask = inflate_grid_cells(scanmapper, route_cells, scan_radius)
    scanned_mask[scanmapper.scanmap == OBSTACLE] = False
    return scanned_mask


def get_search_radius_m(scanmapper, search_radius_m=None):
    """Use the ScanMapper scan radius unless a caller overrides it."""
    if search_radius_m is not None:
        return search_radius_m

    return getattr(scanmapper, "scanradius", 0.7)


def safe_free_mask(scanmap, clearance_cells=DEFAULT_CLEARANCE_CELLS):
    """
    Return cells that are free and at least clearance_cells away from obstacles.

    This avoids choosing patrol points right beside obstacles.
    """
    obstacle_mask = scanmap == OBSTACLE
    unsafe = obstacle_mask.copy()

    offsets = []
    for dy in range(-clearance_cells, clearance_cells + 1):
        for dx in range(-clearance_cells, clearance_cells + 1):
            if (dx * dx) + (dy * dy) <= clearance_cells * clearance_cells:
                offsets.append((dy, dx))

    for dy, dx in offsets:
        src_y0 = max(0, -dy)
        src_y1 = min(scanmap.shape[0], scanmap.shape[0] - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(scanmap.shape[1], scanmap.shape[1] - dx)

        dst_y0 = src_y0 + dy
        dst_y1 = src_y1 + dy
        dst_x0 = src_x0 + dx
        dst_x1 = src_x1 + dx

        unsafe[dst_y0:dst_y1, dst_x0:dst_x1] |= obstacle_mask[src_y0:src_y1, src_x0:src_x1]

    return (scanmap == FREE) & (~unsafe)


def generate_safe_drone_start_zones(
    scanmapper,
    drone_ids,
    min_distance_m=1.2,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    random_seed=None,
    max_attempts=2000,
    territory_masks=None,
):
    """
    Randomly choose one safe start zone per drone.

    A safe start zone is on free space, away from obstacles, and at least
    min_distance_m away from every other generated start zone.
    """
    rng = random.Random(random_seed)
    safe_mask = safe_free_mask(scanmapper.scanmap, clearance_cells)
    safe_cells = np.argwhere(safe_mask)
    if len(safe_cells) < len(drone_ids):
        raise ValueError("Not enough safe free cells to generate drone start zones.")

    candidates = [
        (grid_yx_to_world_xy(scanmapper, int(y), int(x)), int(x))
        for y, x in safe_cells
    ]

    starts = {}
    for idx, drone_id in enumerate(drone_ids):
        if territory_masks is not None and drone_id in territory_masks:
            band_mask = territory_masks[drone_id] & safe_mask
        else:
            band_width = max(1, scanmapper.width / len(drone_ids))
            band_min = int(idx * band_width)
            band_max = int((idx + 1) * band_width)
            band_mask = np.zeros(scanmapper.scanmap.shape, dtype=bool)
            band_mask[:, band_min:band_max] = safe_mask[:, band_min:band_max]

        band_candidates = []
        for component in connected_components(band_mask):
            center_y = sum(y for y, _ in component) / len(component)
            center_x = sum(x for _, x in component) / len(component)
            component_candidates = sorted(
                (
                    grid_yx_to_world_xy(scanmapper, y, x)
                    for y, x in component
                ),
                key=lambda point: (
                    (point[0] - (center_x * scanmapper.resolution)) ** 2
                    + (point[1] - (center_y * scanmapper.resolution)) ** 2
                ),
            )
            band_candidates.extend(component_candidates)

        for point_xy in band_candidates:
            if all(
                math.hypot(point_xy[0] - other[0], point_xy[1] - other[1]) >= min_distance_m
                for other in starts.values()
            ):
                starts[drone_id] = point_xy
                break

    if len(starts) == len(drone_ids):
        return starts

    point_candidates = [point_xy for point_xy, _ in candidates]
    for _ in range(max_attempts):
        starts = {}
        shuffled = point_candidates.copy()
        rng.shuffle(shuffled)

        for point_xy in shuffled:
            if all(
                math.hypot(point_xy[0] - other[0], point_xy[1] - other[1]) >= min_distance_m
                for other in starts.values()
            ):
                drone_id = drone_ids[len(starts)]
                starts[drone_id] = point_xy
                if len(starts) == len(drone_ids):
                    return starts

    raise ValueError("Could not generate safe drone start zones with the requested spacing.")


def generate_candidate_patrol_points(scanmapper, spacing_m=0.8, clearance_cells=DEFAULT_CLEARANCE_CELLS):
    """
    Sample safe patrol candidates across the whole free map.

    Returns:
        list of (x=east, y=north)
    """
    step = max(1, int(round(spacing_m / scanmapper.resolution)))
    safe_mask = safe_free_mask(scanmapper.scanmap, clearance_cells)

    candidates = []
    for y in range(0, scanmapper.height, step):
        for x in range(0, scanmapper.width, step):
            if safe_mask[y, x]:
                candidates.append(grid_yx_to_world_xy(scanmapper, y, x))

    return candidates


def coverage_cells_for_point(scanmapper, point_xy, search_radius_m):
    """
    Return covered free cells as a set of (y, x) grid cells.

    This approximates camera/search coverage as a circle around the drone.
    """
    center_x, center_y = world_xy_to_grid_xy(scanmapper, point_xy[0], point_xy[1])
    radius_cells = max(1, int(round(search_radius_m / scanmapper.resolution)))

    covered = set()
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if (dx * dx) + (dy * dy) > radius_cells * radius_cells:
                continue

            x = center_x + dx
            y = center_y + dy

            if 0 <= y < scanmapper.height and 0 <= x < scanmapper.width:
                if scanmapper.scanmap[y, x] == FREE:
                    covered.add((y, x))

    return covered


def select_patrol_points_by_coverage(
    scanmapper,
    candidate_points_xy,
    search_radius_m=0.7,
    max_points=18,
    min_new_cells=8,
):
    """
    Greedily choose patrol points that cover the most new free cells.

    This is an approximate set-cover optimizer:
    - exact optimal set cover is expensive
    - greedy max-new-coverage is simple and effective
    """
    candidate_coverages = [
        (point_xy, coverage_cells_for_point(scanmapper, point_xy, search_radius_m))
        for point_xy in candidate_points_xy
    ]

    selected = []
    covered = set()

    for _ in range(max_points):
        best_point = None
        best_new_cells = set()

        for point_xy, coverage in candidate_coverages:
            if point_xy in selected:
                continue

            new_cells = coverage - covered
            if len(new_cells) > len(best_new_cells):
                best_point = point_xy
                best_new_cells = new_cells

        if best_point is None or len(best_new_cells) < min_new_cells:
            break

        selected.append(best_point)
        covered |= best_new_cells

    return selected, covered


def obstacle_edge_score_for_point(scanmapper, point_xy, obstacle_radius_m=1.0):
    """
    Score a patrol point by how much obstacle edge it can watch.

    Higher score means the point is near more obstacle cells. This is useful
    for monitoring corridors, corners, and hiding spots around obstacles.
    """
    center_x, center_y = world_xy_to_grid_xy(scanmapper, point_xy[0], point_xy[1])
    radius_cells = max(1, int(round(obstacle_radius_m / scanmapper.resolution)))

    score = 0.0
    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            dist_sq = (dx * dx) + (dy * dy)
            if dist_sq == 0 or dist_sq > radius_cells * radius_cells:
                continue

            x = center_x + dx
            y = center_y + dy

            if 0 <= y < scanmapper.height and 0 <= x < scanmapper.width:
                if scanmapper.scanmap[y, x] == OBSTACLE:
                    score += 1.0 / math.sqrt(dist_sq)

    return score


def select_patrol_points_by_weighted_score(
    scanmapper,
    candidate_points_xy,
    search_radius_m=0.7,
    max_points=18,
    coverage_weight=1.0,
    obstacle_weight=0.0,
    obstacle_radius_m=1.0,
    min_score=1.0,
    target_coverage_cells=None,
):
    """
    Greedily choose patrol points using coverage and obstacle-edge scoring.

    frontage mode:
        high coverage_weight, low obstacle_weight

    obstacle mode:
        low coverage_weight, high obstacle_weight

    hybrid mode:
        medium/high values for both
    """
    candidate_data = []
    obstacle_scores = []

    for point_xy in candidate_points_xy:
        coverage = coverage_cells_for_point(scanmapper, point_xy, search_radius_m)
        obstacle_score = obstacle_edge_score_for_point(
            scanmapper,
            point_xy,
            obstacle_radius_m=obstacle_radius_m,
        )
        candidate_data.append((point_xy, coverage, obstacle_score))
        obstacle_scores.append(obstacle_score)

    max_obstacle_score = max(obstacle_scores) if obstacle_scores else 0.0
    max_coverage_size = max((len(data[1]) for data in candidate_data), default=1)

    selected = []
    covered = set()

    for _ in range(max_points):
        if target_coverage_cells is not None and len(covered) >= target_coverage_cells:
            break

        best_point = None
        best_new_cells = set()
        best_score = 0.0

        for point_xy, coverage, obstacle_score in candidate_data:
            if point_xy in selected:
                continue

            new_cells = coverage - covered
            normalized_obstacle_score = (
                obstacle_score / max_obstacle_score
                if max_obstacle_score > 0.0
                else 0.0
            )
            obstacle_bonus = normalized_obstacle_score * max_coverage_size
            score = (coverage_weight * len(new_cells)) + (obstacle_weight * obstacle_bonus)

            if score > best_score:
                best_point = point_xy
                best_new_cells = new_cells
                best_score = score

        if best_point is None or best_score < min_score:
            break

        selected.append(best_point)
        covered |= best_new_cells

    return selected, covered


def farthest_first_seeds(points_xy, drone_count):
    """Pick spread-out seed points for clustering patrol points."""
    if not points_xy:
        return []

    points = list(points_xy)
    seeds = [points.pop(0)]

    while points and len(seeds) < drone_count:
        best_idx = max(
            range(len(points)),
            key=lambda idx: min(
                math.hypot(points[idx][0] - seed[0], points[idx][1] - seed[1])
                for seed in seeds
            ),
        )
        seeds.append(points.pop(best_idx))

    return seeds


def partition_points_by_distance(points_xy, drone_ids, drone_start_xy=None, balance=True):
    """
    Assign patrol points to drones.

    If drone_start_xy is provided, points go to nearest drone start.
    Otherwise, use farthest-first seeds to make spatial clusters.
    """
    assignments = {drone_id: [] for drone_id in drone_ids}

    if not points_xy:
        return assignments

    if drone_start_xy:
        centers = {drone_id: drone_start_xy[drone_id] for drone_id in drone_ids}
    else:
        seeds = farthest_first_seeds(points_xy, len(drone_ids))
        centers = {
            drone_id: seeds[min(i, len(seeds) - 1)]
            for i, drone_id in enumerate(drone_ids)
        }

    if not balance:
        for point in points_xy:
            best_drone = min(
                drone_ids,
                key=lambda drone_id: math.hypot(
                    point[0] - centers[drone_id][0],
                    point[1] - centers[drone_id][1],
                ),
            )
            assignments[best_drone].append(point)
        return assignments

    max_points_per_drone = math.ceil(len(points_xy) / len(drone_ids))

    # Assign constrained points first. A point is constrained when it is much
    # closer to one drone/cluster center than the others.
    ranked_points = []
    for point in points_xy:
        distances = sorted(
            (
                math.hypot(point[0] - centers[drone_id][0], point[1] - centers[drone_id][1]),
                drone_id,
            )
            for drone_id in drone_ids
        )
        gap = distances[1][0] - distances[0][0] if len(distances) > 1 else 0.0
        ranked_points.append((gap, point, distances))

    ranked_points.sort(reverse=True, key=lambda item: item[0])

    for _, point, distances in ranked_points:
        placed = False
        for _, drone_id in distances:
            if len(assignments[drone_id]) < max_points_per_drone:
                assignments[drone_id].append(point)
                placed = True
                break

        if not placed:
            least_loaded = min(drone_ids, key=lambda drone_id: len(assignments[drone_id]))
            assignments[least_loaded].append(point)

    return assignments


def partition_points_by_east_territory(points_xy, drone_ids, drone_start_xy):
    """Assign patrol points to non-overlapping east-position territories."""
    if not points_xy:
        return {drone_id: [] for drone_id in drone_ids}

    ordered_drone_ids = sorted(drone_ids, key=lambda drone_id: drone_start_xy[drone_id][0])
    boundaries = []
    for first_id, second_id in zip(ordered_drone_ids, ordered_drone_ids[1:]):
        boundaries.append((drone_start_xy[first_id][0] + drone_start_xy[second_id][0]) / 2.0)

    assignments = {drone_id: [] for drone_id in drone_ids}
    for point in points_xy:
        territory_idx = 0
        while territory_idx < len(boundaries) and point[0] > boundaries[territory_idx]:
            territory_idx += 1
        assignments[ordered_drone_ids[territory_idx]].append(point)

    empty_drones = [drone_id for drone_id in drone_ids if not assignments[drone_id]]
    while empty_drones:
        donor_id = max(drone_ids, key=lambda drone_id: len(assignments[drone_id]))
        if len(assignments[donor_id]) <= 1:
            break
        receiver_id = empty_drones.pop(0)
        assignments[receiver_id].append(assignments[donor_id].pop())

    return assignments


def build_map_balanced_territory_masks(scanmapper, drone_ids, clearance_cells=MIN_ROUTE_BUFFER_CELLS):
    """
    Create disjoint east-west territory masks balanced by safe free cells.

    Physical start positions are ignored here. The map is split first, then
    each drone gets a generated patrol entry point inside its assigned region.
    """
    ordered_drone_ids = list(drone_ids)
    safe_mask = safe_free_mask(scanmapper.scanmap, clearance_cells)
    column_counts = np.count_nonzero(safe_mask, axis=0)
    total_safe_cells = int(np.sum(column_counts))

    if total_safe_cells == 0:
        raise ValueError("No safe free cells available for swarm patrol territories.")

    boundaries = [0]
    running = 0
    target_per_drone = total_safe_cells / len(ordered_drone_ids)
    next_target = target_per_drone

    for x, count in enumerate(column_counts):
        running += int(count)
        remaining_columns = scanmapper.width - (x + 1)
        remaining_splits = len(ordered_drone_ids) - len(boundaries)
        if remaining_splits <= 0:
            break

        if running >= next_target and remaining_columns >= remaining_splits:
            boundaries.append(x + 1)
            next_target += target_per_drone

    while len(boundaries) < len(ordered_drone_ids):
        fallback_x = int(scanmapper.width * len(boundaries) / len(ordered_drone_ids))
        boundaries.append(max(boundaries[-1] + 1, min(fallback_x, scanmapper.width - 1)))

    boundaries = boundaries[:len(ordered_drone_ids)] + [scanmapper.width]
    masks = {}

    for idx, drone_id in enumerate(ordered_drone_ids):
        x0 = boundaries[idx]
        x1 = boundaries[idx + 1]

        mask = np.zeros(scanmapper.scanmap.shape, dtype=bool)
        mask[:, x0:x1] = True
        mask[scanmapper.scanmap == OBSTACLE] = False
        masks[drone_id] = mask

    return masks


def generate_patrol_start_zones(
    scanmapper,
    drone_ids,
    territory_masks,
    min_distance_m=1.2,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    random_seed=None,
):
    """Generate one safe patrol entry point inside each territory."""
    starts = generate_safe_drone_start_zones(
        scanmapper=scanmapper,
        drone_ids=drone_ids,
        min_distance_m=min_distance_m,
        clearance_cells=clearance_cells,
        random_seed=random_seed,
        territory_masks=territory_masks,
    )

    missing = [drone_id for drone_id in drone_ids if drone_id not in starts]
    if missing:
        raise ValueError(f"Could not generate patrol starts for: {missing}")

    return starts


def filter_points_by_mask(scanmapper, points_xy, allowed_mask):
    """Keep only points inside allowed_mask."""
    filtered = []
    for point_xy in points_xy:
        grid_x, grid_y = world_xy_to_grid_xy(scanmapper, point_xy[0], point_xy[1])
        if allowed_mask[grid_y, grid_x]:
            filtered.append(point_xy)
    return filtered


def reachable_mask_from_start(scanmapper, start_xy, allowed_mask):
    """Return cells reachable from start_xy while staying inside allowed_mask."""
    start_x, start_y = world_xy_to_grid_xy(scanmapper, start_xy[0], start_xy[1])
    reachable = np.zeros(scanmapper.scanmap.shape, dtype=bool)

    if not allowed_mask[start_y, start_x]:
        return reachable

    stack = [(start_y, start_x)]
    reachable[start_y, start_x] = True

    while stack:
        y, x = stack.pop()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny = y + dy
            nx = x + dx
            if not (0 <= ny < scanmapper.height and 0 <= nx < scanmapper.width):
                continue
            if reachable[ny, nx] or not allowed_mask[ny, nx]:
                continue
            reachable[ny, nx] = True
            stack.append((ny, nx))

    return reachable


def connected_components(mask):
    """Return connected components from a boolean grid mask as lists of (y, x)."""
    visited = np.zeros(mask.shape, dtype=bool)
    components = []

    for start_y, start_x in np.argwhere(mask):
        if visited[start_y, start_x]:
            continue

        component = []
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True

        while stack:
            y, x = stack.pop()
            component.append((y, x))

            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny = y + dy
                nx = x + dx
                if not (0 <= ny < mask.shape[0] and 0 <= nx < mask.shape[1]):
                    continue
                if visited[ny, nx] or not mask[ny, nx]:
                    continue
                visited[ny, nx] = True
                stack.append((ny, nx))

        components.append(component)

    components.sort(reverse=True, key=len)
    return components


def select_balanced_patrol_points_by_territory(
    scanmapper,
    candidate_points_xy,
    drone_ids,
    patrol_start_xy,
    territory_masks,
    search_radius_m,
    max_patrol_points,
    coverage_weight,
    obstacle_weight,
    obstacle_radius_m=1.2,
    min_point_score=1.0,
    route_balance_weight=1.0,
    target_coverage_ratio=None,
):
    """Select a similar patrol-point budget for each drone territory."""
    safety_mask = safe_free_mask(scanmapper.scanmap, MIN_ROUTE_BUFFER_CELLS)
    max_points_per_drone = max(1, math.ceil(max_patrol_points / len(drone_ids)))

    grouped_points = {}
    all_selected = []
    all_covered = set()

    for drone_id in drone_ids:
        planning_reachable_mask = reachable_mask_from_start(
            scanmapper,
            patrol_start_xy[drone_id],
            territory_masks[drone_id] & safety_mask,
        )
        territory_candidates = filter_points_by_mask(
            scanmapper,
            candidate_points_xy,
            planning_reachable_mask,
        )
        target_coverage_cells = None
        if target_coverage_ratio is not None:
            territory_free_cells = int(np.count_nonzero(planning_reachable_mask))
            target_coverage_cells = math.ceil(territory_free_cells * target_coverage_ratio)

        selected, covered = select_patrol_points_by_weighted_score(
            scanmapper=scanmapper,
            candidate_points_xy=territory_candidates,
            search_radius_m=search_radius_m,
            max_points=max_points_per_drone,
            coverage_weight=coverage_weight,
            obstacle_weight=obstacle_weight,
            obstacle_radius_m=obstacle_radius_m,
            min_score=min_point_score,
            target_coverage_cells=target_coverage_cells,
        )

        target_reached = (
            target_coverage_cells is not None
            and len(covered) >= target_coverage_cells
        )
        if route_balance_weight > 0:
            balance_target_count = len(selected) if target_reached else max_points_per_drone
            selected = fill_patrol_points_for_route_balance(
                candidate_points_xy=territory_candidates,
                selected_points_xy=selected,
                start_xy=patrol_start_xy[drone_id],
                target_count=balance_target_count,
            )

        grouped_points[drone_id] = selected
        all_selected.extend(selected)
        all_covered |= covered

    return grouped_points, all_selected, all_covered, territory_masks


def fill_patrol_points_for_route_balance(candidate_points_xy, selected_points_xy, start_xy, target_count):
    """Add safe reachable filler points so each drone has a similar loop budget."""
    if not candidate_points_xy:
        return list(selected_points_xy)

    spread_points = []
    for key_fn in (
        lambda point: point[1],
        lambda point: -point[1],
        lambda point: point[0],
        lambda point: -point[0],
    ):
        point = min(candidate_points_xy, key=key_fn)
        if point not in spread_points:
            spread_points.append(point)

    selected = spread_points.copy()
    for point in selected_points_xy:
        if point not in selected:
            selected.append(point)

    if len(selected) > target_count:
        mandatory = spread_points[:target_count]
        optional = [
            point
            for point in selected
            if point not in mandatory
        ]
        selected = mandatory
        while optional and len(selected) < target_count:
            best_idx = max(
                range(len(optional)),
                key=lambda idx: min(
                    math.hypot(optional[idx][0] - anchor[0], optional[idx][1] - anchor[1])
                    for anchor in selected
                ),
            )
            selected.append(optional.pop(best_idx))
        return selected

    remaining = [
        point
        for point in candidate_points_xy
        if point not in selected
    ]

    while remaining and len(selected) < target_count:
        anchors = selected or [start_xy]
        best_idx = max(
            range(len(remaining)),
            key=lambda idx: min(
                math.hypot(remaining[idx][0] - anchor[0], remaining[idx][1] - anchor[1])
                for anchor in anchors
            ),
        )
        selected.append(remaining.pop(best_idx))

    return selected


def nearest_neighbor_order(points_xy, start_xy=None):
    """Fast route ordering heuristic for patrol points."""
    if not points_xy:
        return []

    remaining = list(points_xy)
    current = start_xy if start_xy is not None else remaining.pop(0)
    ordered = []

    while remaining:
        best_idx = min(
            range(len(remaining)),
            key=lambda idx: math.hypot(remaining[idx][0] - current[0], remaining[idx][1] - current[1]),
        )
        current = remaining.pop(best_idx)
        ordered.append(current)

    return ordered


def two_opt_order(points_xy, passes=2):
    """Small 2-opt route improvement on Euclidean point order."""
    route = list(points_xy)
    if len(route) < 4:
        return route

    def segment_gain(i, j):
        a, b = route[i - 1], route[i]
        c, d = route[j], route[(j + 1) % len(route)]
        before = math.hypot(a[0] - b[0], a[1] - b[1]) + math.hypot(c[0] - d[0], c[1] - d[1])
        after = math.hypot(a[0] - c[0], a[1] - c[1]) + math.hypot(b[0] - d[0], b[1] - d[1])
        return before - after

    for _ in range(passes):
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                if segment_gain(i, j) > 0:
                    route[i : j + 1] = reversed(route[i : j + 1])
                    improved = True
        if not improved:
            break

    return route


def connect_world_points_with_astar(
    scanmapper,
    points_xy,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    loop_route=True,
    extra_blocked_mask=None,
    allow_relaxed_fallback=True,
):
    """
    Connect patrol points with A* and return executable waypoints.
    """
    if len(points_xy) < 2:
        return points_xy

    full_route = [points_xy[0]]
    pairs = list(zip(points_xy, points_xy[1:]))
    if loop_route:
        pairs.append((points_xy[-1], points_xy[0]))

    for start_xy, goal_xy in pairs:
        goal_cell = world_xy_to_grid_xy(scanmapper, goal_xy[0], goal_xy[1])
        segment = plan_path_to_grid_goal(
            scanmapper=scanmapper,
            start_xy=start_xy,
            goal_xy_cell=goal_cell,
            buffer_cells=buffer_cells,
            extra_blocked_mask=extra_blocked_mask,
        )
        if segment is None and extra_blocked_mask is not None and allow_relaxed_fallback:
            segment = plan_path_to_grid_goal(
                scanmapper=scanmapper,
                start_xy=start_xy,
                goal_xy_cell=goal_cell,
                buffer_cells=buffer_cells,
                extra_blocked_mask=None,
            )
        relaxed_buffer_cells = max(MIN_ROUTE_BUFFER_CELLS, buffer_cells // 2)
        if segment is None and relaxed_buffer_cells < buffer_cells:
            segment = plan_path_to_grid_goal(
                scanmapper=scanmapper,
                start_xy=start_xy,
                goal_xy_cell=goal_cell,
                buffer_cells=relaxed_buffer_cells,
                extra_blocked_mask=extra_blocked_mask,
            )

        if segment:
            full_route.extend(segment[1:])

    return full_route


def build_swarm_routes_from_patrol_points(
    scanmapper,
    selected_points,
    covered_cells,
    drone_ids,
    drone_start_xy=None,
    physical_start_xy=None,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    scan_radius_m=None,
    path_separation_m=0.25,
    avoid_route_overlap=True,
    grouped_points_override=None,
    territory_masks=None,
    equalize_loop_lengths=True,
    loop_route=True,
    two_opt_passes=2,
    target_coverage_ratio=None,
):
    """Split selected patrol points across drones and connect them with A*."""
    if grouped_points_override is not None:
        grouped_points = grouped_points_override
    elif avoid_route_overlap and drone_start_xy:
        grouped_points = partition_points_by_east_territory(
            points_xy=selected_points,
            drone_ids=drone_ids,
            drone_start_xy=drone_start_xy,
        )
    else:
        grouped_points = partition_points_by_distance(
            points_xy=selected_points,
            drone_ids=drone_ids,
            drone_start_xy=drone_start_xy,
        )

    routes = {}
    reserved_route_mask = np.zeros(scanmapper.scanmap.shape, dtype=bool)
    scanned_masks = {}

    for drone_id, patrol_points in grouped_points.items():
        start_xy = drone_start_xy.get(drone_id) if drone_start_xy else None
        ordered = nearest_neighbor_order(patrol_points, start_xy=start_xy)
        ordered = two_opt_order(ordered, passes=two_opt_passes)

        if start_xy and ordered:
            route_points = [start_xy] + ordered
        else:
            route_points = ordered

        route_blocked_mask = reserved_route_mask.copy() if avoid_route_overlap else None
        if territory_masks is not None and drone_id in territory_masks:
            territory_blocked = ~territory_masks[drone_id]
            if route_blocked_mask is None:
                route_blocked_mask = territory_blocked.copy()
            else:
                route_blocked_mask |= territory_blocked

        executable_route = connect_world_points_with_astar(
            scanmapper=scanmapper,
            points_xy=route_points,
            buffer_cells=buffer_cells,
            loop_route=loop_route,
            extra_blocked_mask=route_blocked_mask,
            allow_relaxed_fallback=territory_masks is None,
        )

        route_cells = route_center_cells(scanmapper, executable_route)
        scanned_mask = trace_route_scan_mask(scanmapper, executable_route, scan_radius_m)
        scanned_masks[drone_id] = scanned_mask

        if avoid_route_overlap:
            reserved_route_mask |= inflate_grid_cells(
                scanmapper,
                route_cells,
                path_separation_m,
            )
            reserved_route_mask[scanmapper.scanmap == OBSTACLE] = False

        routes[drone_id] = {
            "patrol_points": ordered,
            "waypoints": executable_route,
            "length_m": route_length(executable_route),
            "route_cells": route_cells,
            "scanned_cells": int(np.count_nonzero(scanned_mask)),
        }

    if equalize_loop_lengths and routes:
        target_length = max(route["length_m"] for route in routes.values())
        for drone_id, route in routes.items():
            if route["length_m"] <= 0 or route["length_m"] >= target_length * 0.75:
                continue

            repeated_waypoints = list(route["waypoints"])
            base_loop = route["waypoints"][1:] if len(route["waypoints"]) > 1 else []
            while base_loop and route_length(repeated_waypoints) < target_length * 0.9:
                repeated_waypoints.extend(base_loop)

            route_cells = route_center_cells(scanmapper, repeated_waypoints)
            scanned_mask = trace_route_scan_mask(scanmapper, repeated_waypoints, scan_radius_m)
            scanned_masks[drone_id] = scanned_mask
            route["waypoints"] = repeated_waypoints
            route["length_m"] = route_length(repeated_waypoints)
            route["route_cells"] = route_cells
            route["scanned_cells"] = int(np.count_nonzero(scanned_mask))

    total_free = int(np.count_nonzero(scanmapper.scanmap == FREE))
    traced_coverage_mask = np.zeros(scanmapper.scanmap.shape, dtype=bool)
    overlap_mask = np.zeros(scanmapper.scanmap.shape, dtype=bool)
    for scanned_mask in scanned_masks.values():
        overlap_mask |= traced_coverage_mask & scanned_mask
        traced_coverage_mask |= scanned_mask

    coverage_ratio = int(np.count_nonzero(traced_coverage_mask)) / total_free if total_free else 0.0

    return {
        "routes": routes,
        "selected_points": selected_points,
        "covered_cells": len(covered_cells),
        "traced_covered_cells": int(np.count_nonzero(traced_coverage_mask)),
        "total_free_cells": total_free,
        "coverage_ratio": coverage_ratio,
        "target_coverage_ratio": target_coverage_ratio,
        "scan_radius_m": get_search_radius_m(scanmapper, scan_radius_m),
        "drone_start_xy": drone_start_xy,
        "patrol_start_xy": drone_start_xy,
        "physical_start_xy": physical_start_xy,
        "scanned_masks": scanned_masks,
        "combined_scanned_mask": traced_coverage_mask,
        "overlap_cells": int(np.count_nonzero(overlap_mask)),
    }


def optimize_tunable_swarm_patrol_routes(
    scanmapper,
    drone_ids,
    drone_start_xy=None,
    search_radius_m=None,
    coverage_weight=0.75,
    obstacle_weight=0.55,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=24,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    obstacle_radius_m=1.2,
    min_point_score=1.0,
    min_start_distance_m=1.2,
    random_seed=None,
    path_separation_m=0.25,
    avoid_route_overlap=True,
    route_balance_weight=1.0,
    equalize_loop_lengths=True,
    loop_route=True,
    two_opt_passes=2,
):
    """
    Build swarm patrol routes with one tunable scoring model.

    Main knobs:
        coverage_weight: prioritize scanning new free cells.
        obstacle_weight: prioritize obstacle edges, corners, and corridors.
        target_coverage_ratio: desired free-map coverage before using more patrol points.
        max_patrol_points: total patrol points shared across drones.
        candidate_spacing_m: patrol candidate sampling density.
        clearance_cells / buffer_cells: point and path obstacle safety margins.
        path_separation_m: route separation between drones.
    """
    active_search_radius_m = get_search_radius_m(scanmapper, search_radius_m)
    territory_masks = build_map_balanced_territory_masks(
        scanmapper=scanmapper,
        drone_ids=drone_ids,
        clearance_cells=clearance_cells,
    )
    patrol_start_xy = generate_patrol_start_zones(
        scanmapper=scanmapper,
        drone_ids=drone_ids,
        territory_masks=territory_masks,
        min_distance_m=min_start_distance_m,
        clearance_cells=clearance_cells,
        random_seed=random_seed,
    )

    candidates = generate_candidate_patrol_points(
        scanmapper=scanmapper,
        spacing_m=candidate_spacing_m,
        clearance_cells=clearance_cells,
    )

    grouped_points, selected_points, covered_cells, territory_masks = select_balanced_patrol_points_by_territory(
        scanmapper=scanmapper,
        candidate_points_xy=candidates,
        drone_ids=drone_ids,
        patrol_start_xy=patrol_start_xy,
        territory_masks=territory_masks,
        search_radius_m=active_search_radius_m,
        max_patrol_points=max_patrol_points,
        coverage_weight=coverage_weight,
        obstacle_weight=obstacle_weight,
        obstacle_radius_m=obstacle_radius_m,
        min_point_score=min_point_score,
        route_balance_weight=route_balance_weight,
        target_coverage_ratio=target_coverage_ratio,
    )

    return build_swarm_routes_from_patrol_points(
        scanmapper=scanmapper,
        selected_points=selected_points,
        covered_cells=covered_cells,
        drone_ids=drone_ids,
        drone_start_xy=patrol_start_xy,
        physical_start_xy=drone_start_xy,
        buffer_cells=buffer_cells,
        scan_radius_m=active_search_radius_m,
        path_separation_m=path_separation_m,
        avoid_route_overlap=avoid_route_overlap,
        grouped_points_override=grouped_points,
        territory_masks=territory_masks if avoid_route_overlap else None,
        equalize_loop_lengths=equalize_loop_lengths,
        loop_route=loop_route,
        two_opt_passes=two_opt_passes,
        target_coverage_ratio=target_coverage_ratio,
    )


def optimize_swarm_patrol_routes(
    scanmapper,
    drone_ids,
    drone_start_xy=None,
    search_radius_m=None,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=24,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    min_start_distance_m=1.2,
    random_seed=None,
    path_separation_m=0.25,
    avoid_route_overlap=True,
):
    """Preset for open maps: prioritize broad free-space coverage."""
    return optimize_tunable_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=drone_ids,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        coverage_weight=1.0,
        obstacle_weight=0.2,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
        min_start_distance_m=min_start_distance_m,
        random_seed=random_seed,
        path_separation_m=path_separation_m,
        avoid_route_overlap=avoid_route_overlap,
    )


def optimize_obstacle_oriented_patrol_routes(
    scanmapper,
    drone_ids,
    drone_start_xy=None,
    search_radius_m=None,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=33,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    obstacle_radius_m=1.2,
    min_start_distance_m=1.2,
    random_seed=None,
    path_separation_m=0.25,
    avoid_route_overlap=True,
):
    """
    Build patrol routes biased toward obstacle edges and corridors.

    This is useful when the target is likely to move near obstacles or when
    obstacle layout creates natural roads/choke points.
    """
    return optimize_tunable_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=drone_ids,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        coverage_weight=0.50,
        obstacle_weight=0.85,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
        obstacle_radius_m=obstacle_radius_m,
        min_start_distance_m=min_start_distance_m,
        random_seed=random_seed,
        path_separation_m=path_separation_m,
        avoid_route_overlap=avoid_route_overlap,
    )


def optimize_hybrid_swarm_patrol_routes(
    scanmapper,
    drone_ids,
    drone_start_xy=None,
    search_radius_m=None,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=24,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    obstacle_radius_m=1.2,
    min_start_distance_m=1.2,
    random_seed=None,
    path_separation_m=0.25,
    avoid_route_overlap=True,
):
    """
    Build patrol routes that balance free-space coverage and obstacle edges.

    This is the safest competition default when we do not know whether the
    target prefers open space or obstacle-heavy corridors.
    """
    return optimize_tunable_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=drone_ids,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        coverage_weight=0.75,
        obstacle_weight=0.55,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
        obstacle_radius_m=obstacle_radius_m,
        min_start_distance_m=min_start_distance_m,
        random_seed=random_seed,
        path_separation_m=path_separation_m,
        avoid_route_overlap=avoid_route_overlap,
    )


def render_patrol_result_map(scanmapper, result, output_path, use_matplotlib=False):
    """
    Save a virtual map image showing per-drone scanned areas and routes.

    Uses OpenCV by default for fast dry-runs. Set use_matplotlib=True for a
    labelled matplotlib plot when matplotlib is installed and configured.
    """
    drone_colors_rgb = {
        "H1": (255, 80, 80),
        "H2": (80, 180, 255),
        "H3": (80, 220, 120),
    }

    base = np.zeros((scanmapper.height, scanmapper.width, 3), dtype=np.uint8)
    base[scanmapper.scanmap == FREE] = (245, 245, 245)
    base[scanmapper.scanmap == OBSTACLE] = (45, 45, 45)
    base[scanmapper.scanmap == SEARCHED] = (230, 230, 160)

    overlay = base.astype(np.float32)
    for drone_id, scanned_mask in result.get("scanned_masks", {}).items():
        color = np.array(drone_colors_rgb.get(drone_id, (180, 120, 255)), dtype=np.float32)
        overlay[scanned_mask] = (overlay[scanned_mask] * 0.45) + (color * 0.55)

    image = np.clip(overlay, 0, 255).astype(np.uint8)

    if use_matplotlib:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 12))
        ax.imshow(image, origin="lower")
        ax.set_title("Swarm Patrol Search Areas")
        ax.set_xlabel("East cells")
        ax.set_ylabel("North cells")

        for drone_id, route in result["routes"].items():
            waypoints = route["waypoints"]
            if not waypoints:
                continue

            xs = []
            ys = []
            for x_east, y_north in waypoints:
                grid_x, grid_y = world_xy_to_grid_xy(scanmapper, x_east, y_north)
                xs.append(grid_x)
                ys.append(grid_y)

            color = np.array(drone_colors_rgb.get(drone_id, (180, 120, 255))) / 255.0
            ax.plot(xs, ys, color=color, linewidth=1.8, label=drone_id)

            start_xy = result.get("drone_start_xy", {}).get(drone_id)
            if start_xy:
                start_x, start_y = world_xy_to_grid_xy(scanmapper, start_xy[0], start_xy[1])
                ax.scatter([start_x], [start_y], color=[color], edgecolor="black", s=60)
                ax.text(start_x + 1, start_y + 1, drone_id, color=color, weight="bold")

        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return output_path

    import cv2

    bgr = cv2.cvtColor(np.flipud(image), cv2.COLOR_RGB2BGR)
    scale = 8
    bgr = cv2.resize(bgr, (scanmapper.width * scale, scanmapper.height * scale), interpolation=cv2.INTER_NEAREST)

    for drone_id, route in result["routes"].items():
        color_rgb = drone_colors_rgb.get(drone_id, (180, 120, 255))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        points = []
        for x_east, y_north in route["waypoints"]:
            grid_x, grid_y = world_xy_to_grid_xy(scanmapper, x_east, y_north)
            px = grid_x * scale
            py = (scanmapper.height - 1 - grid_y) * scale
            points.append((px, py))

        for start, end in zip(points, points[1:]):
            cv2.line(bgr, start, end, color_bgr, 2)

        start_xy = result.get("drone_start_xy", {}).get(drone_id)
        if start_xy:
            start_x, start_y = world_xy_to_grid_xy(scanmapper, start_xy[0], start_xy[1])
            px = start_x * scale
            py = (scanmapper.height - 1 - start_y) * scale
            cv2.circle(bgr, (px, py), 5, color_bgr, -1)
            cv2.putText(bgr, drone_id, (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1)

    cv2.imwrite(output_path, bgr)
    return output_path
