import argparse
import os
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from optimized_swarm_path_planner import (
    DEFAULT_BUFFER_CELLS,
    DEFAULT_CLEARANCE_CELLS,
    FREE,
    OBSTACLE,
    optimize_hybrid_swarm_patrol_routes,
    optimize_obstacle_oriented_patrol_routes,
    optimize_swarm_patrol_routes,
    trace_route_scan_mask,
    world_xy_to_grid_xy,
)


class FakeScanMapper:
    def __init__(
        self,
        heightcells_NORTHLENGTH=110,
        widthcells_EASTLENGTH=55,
        metrespercell=0.1,
        ScanmapOriginOffset_Exm=0.0,
        ScanmapOriginOffset_Nym=0.0,
        scanradius=0.7,
    ):
        self.height = heightcells_NORTHLENGTH
        self.width = widthcells_EASTLENGTH
        self.resolution = metrespercell
        self.ScanmapOriginOffset_Exm = ScanmapOriginOffset_Exm
        self.ScanmapOriginOffset_Nym = ScanmapOriginOffset_Nym
        self.scanradius = scanradius
        self.scanmap = np.full((self.height, self.width), FREE, dtype=np.int8)

        self.scanmap[0:2, :] = OBSTACLE
        self.scanmap[-2:, :] = OBSTACLE
        self.scanmap[:, 0:2] = OBSTACLE
        self.scanmap[:, -2:] = OBSTACLE

    def worldNE_to_scanmapXY(self, N, E, clamp=True):
        x = round((E - self.ScanmapOriginOffset_Exm) / self.resolution)
        y = round((N - self.ScanmapOriginOffset_Nym) / self.resolution)

        if clamp:
            x = max(0, min(x, self.width - 1))
            y = max(0, min(y, self.height - 1))

        return int(x), int(y)

    def scanmapXY_to_worldNE(self, X, Y):
        north = Y * self.resolution + self.ScanmapOriginOffset_Nym
        east = X * self.resolution + self.ScanmapOriginOffset_Exm
        return north, east


def add_rect_obstacles(scanmapper, rectangles):
    for y0, y1, x0, x1 in rectangles:
        scanmapper.scanmap[y0:y1, x0:x1] = OBSTACLE


def keep_vertical_travel_corridors(scanmapper, corridor_centers=(9, 27, 45), half_width=2):
    for center_x in corridor_centers:
        x0 = max(2, center_x - half_width)
        x1 = min(scanmapper.width - 2, center_x + half_width + 1)
        scanmapper.scanmap[4:-4, x0:x1] = FREE


def make_wall_corridor_map():
    scanmapper = FakeScanMapper()

    scanmapper.scanmap[20:95, 22:26] = OBSTACLE
    scanmapper.scanmap[45:55, 22:26] = FREE

    scanmapper.scanmap[15:85, 38:42] = OBSTACLE
    scanmapper.scanmap[30:40, 38:42] = FREE

    return scanmapper


def make_open_map():
    scanmapper = FakeScanMapper()
    add_rect_obstacles(
        scanmapper,
        [
            (13, 21, 8, 13),
            (25, 34, 33, 39),
            (70, 80, 10, 17),
            (84, 94, 29, 36),
        ],
    )
    return scanmapper


def make_block_cluster_map():
    scanmapper = make_open_map()

    blocks = [
        (18, 34, 10, 18),
        (45, 62, 26, 35),
        (75, 92, 12, 22),
        (72, 90, 38, 47),
        (92, 101, 28, 35),
    ]
    add_rect_obstacles(scanmapper, blocks)
    keep_vertical_travel_corridors(scanmapper, corridor_centers=(8, 27, 46), half_width=2)

    return scanmapper


def make_random_obstacle_map(random_seed=123):
    scanmapper = make_open_map()
    rng = np.random.default_rng(random_seed)

    for _ in range(8):
        height = int(rng.integers(4, 11))
        width = int(rng.integers(3, 7))
        y0 = int(rng.integers(8, scanmapper.height - height - 8))
        x0 = int(rng.integers(6, scanmapper.width - width - 6))
        scanmapper.scanmap[y0:y0 + height, x0:x0 + width] = OBSTACLE

    add_rect_obstacles(
        scanmapper,
        [
            (28, 33, 16, 24),
            (58, 63, 31, 39),
            (88, 93, 14, 22),
        ],
    )
    keep_vertical_travel_corridors(scanmapper, corridor_centers=(9, 27, 45), half_width=3)

    return scanmapper


def plan_routes(scanmapper, mode):
    common_kwargs = {
        "scanmapper": scanmapper,
        "drone_ids": ["H1", "H2", "H3"],
        "search_radius_m": None,
        "max_patrol_points": 24,
        "clearance_cells": 3,
        "buffer_cells": 3,
        "min_start_distance_m": 1.2,
        "random_seed": 42,
    }

    if mode == "frontage":
        return optimize_swarm_patrol_routes(candidate_spacing_m=0.4, **common_kwargs)
    if mode == "obstacle":
        return optimize_obstacle_oriented_patrol_routes(
            candidate_spacing_m=0.4,
            obstacle_radius_m=1.2,
            **common_kwargs,
        )
    if mode == "hybrid":
        return optimize_hybrid_swarm_patrol_routes(
            candidate_spacing_m=0.4,
            obstacle_radius_m=1.2,
            **common_kwargs,
        )

    raise ValueError(f"Unknown mode: {mode}")


def base_rgb_map(scanmapper):
    image = np.zeros((scanmapper.height, scanmapper.width, 3), dtype=np.float32)
    image[scanmapper.scanmap == FREE] = (0.96, 0.96, 0.96)
    image[scanmapper.scanmap == OBSTACLE] = (0.12, 0.12, 0.12)
    return image


def route_prefix(route, step):
    if not route:
        return []
    return route[: min(step + 1, len(route))]


def animate(scanmapper, result, delay_s=0.08):
    colors = {
        "H1": np.array([1.0, 0.25, 0.25]),
        "H2": np.array([0.2, 0.55, 1.0]),
        "H3": np.array([0.25, 0.9, 0.35]),
    }

    routes = {
        drone_id: route["waypoints"]
        for drone_id, route in result["routes"].items()
    }
    max_steps = max(len(route) for route in routes.values())

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 11))

    for step in range(max_steps):
        frame = base_rgb_map(scanmapper)

        for drone_id, route in routes.items():
            partial_route = route_prefix(route, step)
            if not partial_route:
                continue

            scan_mask = trace_route_scan_mask(scanmapper, partial_route, scanmapper.scanradius)
            color = colors[drone_id]
            frame[scan_mask] = (frame[scan_mask] * 0.45) + (color * 0.55)

        ax.clear()
        ax.imshow(frame, origin="lower")
        ax.set_title(f"Swarm patrol search build-up | step {step + 1}/{max_steps}")
        ax.set_xlabel("East cells")
        ax.set_ylabel("North cells")

        for drone_id, route in routes.items():
            partial_route = route_prefix(route, step)
            if not partial_route:
                continue

            patrol_xs = []
            patrol_ys = []
            for x_east, y_north in result["routes"][drone_id]["patrol_points"]:
                grid_x, grid_y = world_xy_to_grid_xy(scanmapper, x_east, y_north)
                patrol_xs.append(grid_x)
                patrol_ys.append(grid_y)
            ax.scatter(
                patrol_xs,
                patrol_ys,
                marker="D",
                color=colors[drone_id],
                edgecolor="black",
                s=35,
                alpha=0.9,
            )

            xs = []
            ys = []
            all_waypoint_xs = []
            all_waypoint_ys = []
            for x_east, y_north in route:
                grid_x, grid_y = world_xy_to_grid_xy(scanmapper, x_east, y_north)
                all_waypoint_xs.append(grid_x)
                all_waypoint_ys.append(grid_y)
            ax.scatter(
                all_waypoint_xs,
                all_waypoint_ys,
                marker=".",
                color=colors[drone_id],
                s=10,
                alpha=0.35,
            )

            for x_east, y_north in partial_route:
                grid_x, grid_y = world_xy_to_grid_xy(scanmapper, x_east, y_north)
                xs.append(grid_x)
                ys.append(grid_y)

            ax.plot(xs, ys, color=colors[drone_id], linewidth=2.0, label=drone_id)
            ax.scatter(xs[-1], ys[-1], color=colors[drone_id], edgecolor="black", s=45)
            ax.text(xs[-1] + 1, ys[-1] + 1, drone_id, color=colors[drone_id], weight="bold")

        ax.legend(loc="upper right")
        ax.text(
            1,
            1,
            "diamonds=patrol points | small dots=A* waypoints",
            color="black",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(delay_s)

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["open", "wall_corridors", "block_clusters", "random_obstacles"],
        default="wall_corridors",
    )
    parser.add_argument(
        "--mode",
        choices=["frontage", "obstacle", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--delay", type=float, default=0.08)
    args = parser.parse_args()

    scenario_factory = {
        "open": make_open_map,
        "wall_corridors": make_wall_corridor_map,
        "block_clusters": make_block_cluster_map,
        "random_obstacles": make_random_obstacle_map,
    }[args.scenario]

    scanmapper = scenario_factory()
    result = plan_routes(scanmapper, args.mode)

    print(f"scenario={args.scenario}")
    print(f"mode={args.mode}")
    print(f"coverage={result['coverage_ratio']:.1%}")
    print(f"starts={result['drone_start_xy']}")
    for drone_id, route in result["routes"].items():
        print(
            f"{drone_id}: waypoints={len(route['waypoints'])} "
            f"patrol_points={len(route['patrol_points'])} "
            f"length={route['length_m']:.2f}m"
        )

    animate(scanmapper, result, delay_s=args.delay)


if __name__ == "__main__":
    main()
