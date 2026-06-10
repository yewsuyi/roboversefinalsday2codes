import os

import numpy as np

from optimized_swarm_path_planner import (
    DEFAULT_BUFFER_CELLS,
    DEFAULT_CLEARANCE_CELLS,
    FREE,
    MIN_ROUTE_BUFFER_CELLS,
    OBSTACLE,
    generate_safe_drone_start_zones,
    obstacle_edge_score_for_point,
    optimize_hybrid_swarm_patrol_routes,
    optimize_obstacle_oriented_patrol_routes,
    optimize_swarm_patrol_routes,
    optimize_tunable_swarm_patrol_routes,
    plan_path_to_grid_goal,
    render_patrol_result_map,
    route_center_cells,
    route_length,
    scanmap_to_obstaclemap,
    world_xy_to_grid_xy,
)


class FakeScanMapper:
    def __init__(
        self,
        heightcells_NORTHLENGTH,
        widthcells_EASTLENGTH,
        metrespercell,
        ScanmapOriginOffset_Exm,
        ScanmapOriginOffset_Nym,
        scanradius,
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
    """Keep the stress maps connected without making them visually empty."""
    for center_x in corridor_centers:
        x0 = max(2, center_x - half_width)
        x1 = min(scanmapper.width - 2, center_x + half_width + 1)
        scanmapper.scanmap[4:-4, x0:x1] = FREE


def make_test_scanmapper():
    scanmapper = FakeScanMapper(
        heightcells_NORTHLENGTH=110,
        widthcells_EASTLENGTH=55,
        metrespercell=0.1,
        ScanmapOriginOffset_Exm=0.0,
        ScanmapOriginOffset_Nym=0.0,
        scanradius=0.7,
    )

    # Add two obstacle walls with gaps.
    scanmapper.scanmap[20:95, 22:26] = OBSTACLE
    scanmapper.scanmap[45:55, 22:26] = FREE

    scanmapper.scanmap[15:85, 38:42] = OBSTACLE
    scanmapper.scanmap[30:40, 38:42] = FREE

    return scanmapper


def make_open_scanmapper():
    scanmapper = FakeScanMapper(
        heightcells_NORTHLENGTH=110,
        widthcells_EASTLENGTH=55,
        metrespercell=0.1,
        ScanmapOriginOffset_Exm=0.0,
        ScanmapOriginOffset_Nym=0.0,
        scanradius=0.7,
    )

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


def make_block_cluster_scanmapper():
    scanmapper = make_open_scanmapper()

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


def make_horizontal_wall_scanmapper():
    scanmapper = make_open_scanmapper()

    # Irregular horizontal walls with staggered gaps. This stresses
    # north/south movement while leaving enough space for high coverage.
    walls = [
        (24, 27, 4, 50, [(7, 20), (26, 39), (43, 50)]),
        (67, 70, 5, 52, [(4, 18), (25, 38), (42, 51)]),
    ]

    for y0, y1, x0, x1, gap_ranges in walls:
        scanmapper.scanmap[y0:y1, x0:x1] = OBSTACLE
        for gap_x0, gap_x1 in gap_ranges:
            scanmapper.scanmap[y0:y1, gap_x0:gap_x1] = FREE

    shelves = [
        (43, 48, 18, 24),
        (88, 93, 35, 42),
    ]
    for y0, y1, x0, x1 in shelves:
        scanmapper.scanmap[y0:y1, x0:x1] = OBSTACLE

    return scanmapper


def make_random_obstacle_scanmapper(random_seed=123):
    scanmapper = make_open_scanmapper()
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


def assert_waypoints_are_safe(scanmapper, waypoints, label):
    obstaclemap = scanmap_to_obstaclemap(scanmapper.scanmap)

    for point in waypoints:
        x, y = scanmapper.worldNE_to_scanmapXY(point[1], point[0])
        assert obstaclemap[y, x] == 0, f"{label} waypoint {point} is inside obstacle"


def assert_route_obstacle_clearance(scanmapper, result, min_clearance_cells=MIN_ROUTE_BUFFER_CELLS):
    obstacle_cells = np.argwhere(scanmapper.scanmap == OBSTACLE)
    if len(obstacle_cells) == 0:
        return

    for drone_id, route in result["routes"].items():
        for route_y, route_x in route_center_cells(scanmapper, route["waypoints"]):
            distances_sq = (
                ((obstacle_cells[:, 0] - route_y) ** 2)
                + ((obstacle_cells[:, 1] - route_x) ** 2)
            )
            nearest = np.sqrt(np.min(distances_sq))
            assert nearest >= min_clearance_cells, (
                f"{drone_id} route is {nearest:.1f} cells from obstacle; "
                f"minimum is {min_clearance_cells}"
            )


def assert_start_zones_are_valid(scanmapper, starts_xy, min_distance_m):
    obstaclemap = scanmap_to_obstaclemap(scanmapper.scanmap)
    starts = list(starts_xy.items())

    for drone_id, point in starts:
        x, y = scanmapper.worldNE_to_scanmapXY(point[1], point[0])
        assert obstaclemap[y, x] == 0, f"{drone_id} start zone {point} is inside obstacle"

    for i in range(len(starts)):
        for j in range(i + 1, len(starts)):
            first_id, first = starts[i]
            second_id, second = starts[j]
            distance = np.hypot(first[0] - second[0], first[1] - second[1])
            assert distance >= min_distance_m, f"{first_id} and {second_id} starts too close"


def count_route_centerline_overlap(scanmapper, result):
    occupied = set()
    overlap_count = 0

    for drone_id, route in result["routes"].items():
        route_cells = set(route_center_cells(scanmapper, route["waypoints"]))
        repeated = occupied & route_cells
        overlap_count += len(repeated)
        occupied |= route_cells

    return overlap_count


def assert_route_centerlines_do_not_overlap(scanmapper, result, max_overlap_cells=0):
    overlap_count = count_route_centerline_overlap(scanmapper, result)
    assert overlap_count <= max_overlap_cells, (
        f"route centerlines overlap by {overlap_count} cells; "
        f"allowed {max_overlap_cells}"
    )


def assert_loop_lengths_are_balanced(result, max_spread_m=12.0, max_ratio=1.8):
    lengths = [
        route["length_m"]
        for route in result["routes"].values()
        if route["length_m"] > 0
    ]
    assert len(lengths) == len(result["routes"]), "one or more routes have zero length"

    shortest = min(lengths)
    longest = max(lengths)
    assert longest - shortest <= max_spread_m, (
        f"route length spread too large: shortest={shortest:.2f}m longest={longest:.2f}m"
    )
    assert longest / shortest <= max_ratio, (
        f"route length ratio too large: shortest={shortest:.2f}m longest={longest:.2f}m"
    )


def assert_planner_result_integrated(
    scanmapper,
    result,
    label,
    min_coverage_ratio=0.80,
    max_route_overlap_cells=0,
):
    assert result["scan_radius_m"] == scanmapper.scanradius, f"{label} did not use scanmapper.scanradius"
    assert result["drone_start_xy"], f"{label} did not return generated drone starts"
    assert_start_zones_are_valid(scanmapper, result["drone_start_xy"], min_distance_m=1.2)
    assert_route_centerlines_do_not_overlap(
        scanmapper,
        result,
        max_overlap_cells=max_route_overlap_cells,
    )
    assert_route_obstacle_clearance(scanmapper, result)
    assert result["combined_scanned_mask"].shape == scanmapper.scanmap.shape
    assert result["traced_covered_cells"] == int(np.count_nonzero(result["combined_scanned_mask"]))
    assert result["coverage_ratio"] >= min_coverage_ratio, f"{label} coverage ratio too low"
    assert_loop_lengths_are_balanced(result)

    for drone_id, route in result["routes"].items():
        assert route["patrol_points"], f"{label} {drone_id} has no patrol points"
        assert route["waypoints"], f"{label} {drone_id} has no executable route"
        assert route["length_m"] > 0, f"{label} {drone_id} route has zero length"
        assert route["route_cells"], f"{label} {drone_id} has no traced route cells"
        assert route["scanned_cells"] > 0, f"{label} {drone_id} scanned zero cells"
        assert_waypoints_are_safe(scanmapper, route["waypoints"], f"{label} {drone_id}")


def test_astar_route_on_cluttered_map():
    scanmapper = make_open_scanmapper()
    start_xy = (1.3, 5.5)
    goal_xy = (4.5, 1.5)
    goal_cell = world_xy_to_grid_xy(scanmapper, goal_xy[0], goal_xy[1])

    route = plan_path_to_grid_goal(
        scanmapper=scanmapper,
        start_xy=start_xy,
        goal_xy_cell=goal_cell,
        buffer_cells=MIN_ROUTE_BUFFER_CELLS,
    )

    assert route and len(route) > 1, "A* route did not move on cluttered map"
    assert_waypoints_are_safe(scanmapper, route, "cluttered A*")

    print("\nCluttered-map A* route:")
    print(f"  waypoints={len(route)}")
    print(f"  first={route[0]}")
    print(f"  last={route[-1]}")


def test_optimized_swarm_patrol_routes():
    scanmapper = make_test_scanmapper()

    result = optimize_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=["H1", "H2", "H3"],
        search_radius_m=None,
        candidate_spacing_m=0.4,
        max_patrol_points=24,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        min_start_distance_m=1.2,
        random_seed=7,
    )

    assert result["scan_radius_m"] == scanmapper.scanradius
    assert_start_zones_are_valid(scanmapper, result["drone_start_xy"], min_distance_m=1.2)
    assert_route_centerlines_do_not_overlap(scanmapper, result)
    assert_loop_lengths_are_balanced(result)

    print("\nOptimized swarm patrol routes:")
    print(
        f"  selected_points={len(result['selected_points'])} "
        f"coverage={result['coverage_ratio']:.1%}"
    )
    print(f"  generated_starts={result['drone_start_xy']}")
    print(f"  scanned_overlap_cells={result['overlap_cells']}")

    for drone_id, route in result["routes"].items():
        assert route["patrol_points"], f"{drone_id} has no patrol points"
        assert route["waypoints"], f"{drone_id} has no executable route"
        assert route["length_m"] > 0, f"{drone_id} route has zero length"
        assert_waypoints_are_safe(scanmapper, route["waypoints"], drone_id)

        print(
            f"  {drone_id}: "
            f"patrol_points={len(route['patrol_points'])} "
            f"waypoints={len(route['waypoints'])} "
            f"length={route['length_m']:.2f}m"
        )

    patrol_counts = [
        len(route["patrol_points"])
        for route in result["routes"].values()
    ]

    assert max(patrol_counts) - min(patrol_counts) <= 8, "patrol points are too imbalanced"
    assert result["coverage_ratio"] >= 0.80, "coverage ratio too low for test map"
    assert sum(route_length(route["waypoints"]) for route in result["routes"].values()) > 0


def assert_swarm_result_is_valid(scanmapper, result, label):
    print(f"\n{label}:")
    print(
        f"  selected_points={len(result['selected_points'])} "
        f"coverage={result['coverage_ratio']:.1%}"
    )

    patrol_counts = []
    for drone_id, route in result["routes"].items():
        assert route["patrol_points"], f"{label} {drone_id} has no patrol points"
        assert route["waypoints"], f"{label} {drone_id} has no executable route"
        assert route["length_m"] > 0, f"{label} {drone_id} route has zero length"
        assert_waypoints_are_safe(scanmapper, route["waypoints"], f"{label} {drone_id}")

        patrol_counts.append(len(route["patrol_points"]))
        print(
            f"  {drone_id}: "
            f"patrol_points={len(route['patrol_points'])} "
            f"waypoints={len(route['waypoints'])} "
            f"length={route['length_m']:.2f}m"
        )

    assert max(patrol_counts) - min(patrol_counts) <= 8, f"{label} patrol points are too imbalanced"
    assert result["coverage_ratio"] >= 0.80, f"{label} coverage ratio too low for test map"


def test_obstacle_and_hybrid_patrol_routes():
    obstacle_scanmapper = make_test_scanmapper()
    obstacle_result = optimize_obstacle_oriented_patrol_routes(
        scanmapper=obstacle_scanmapper,
        drone_ids=["H1", "H2", "H3"],
        search_radius_m=None,
        candidate_spacing_m=0.4,
        max_patrol_points=33,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        obstacle_radius_m=1.2,
        random_seed=11,
    )
    assert obstacle_result["scan_radius_m"] == obstacle_scanmapper.scanradius
    assert_start_zones_are_valid(obstacle_scanmapper, obstacle_result["drone_start_xy"], min_distance_m=1.2)
    assert_route_centerlines_do_not_overlap(obstacle_scanmapper, obstacle_result)
    assert_loop_lengths_are_balanced(obstacle_result)
    assert_swarm_result_is_valid(obstacle_scanmapper, obstacle_result, "Obstacle-oriented patrol routes")

    frontage_scanmapper = make_test_scanmapper()
    frontage_result = optimize_swarm_patrol_routes(
        scanmapper=frontage_scanmapper,
        drone_ids=["H1", "H2", "H3"],
        search_radius_m=None,
        candidate_spacing_m=0.4,
        max_patrol_points=24,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        random_seed=11,
    )

    obstacle_score = sum(
        obstacle_edge_score_for_point(obstacle_scanmapper, point)
        for point in obstacle_result["selected_points"]
    )
    frontage_score = sum(
        obstacle_edge_score_for_point(frontage_scanmapper, point)
        for point in frontage_result["selected_points"]
    )
    print(f"  obstacle_edge_score={obstacle_score:.1f} frontage_edge_score={frontage_score:.1f}")

    hybrid_scanmapper = make_test_scanmapper()
    hybrid_result = optimize_hybrid_swarm_patrol_routes(
        scanmapper=hybrid_scanmapper,
        drone_ids=["H1", "H2", "H3"],
        search_radius_m=None,
        candidate_spacing_m=0.4,
        max_patrol_points=24,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        obstacle_radius_m=1.2,
        random_seed=13,
    )
    assert hybrid_result["scan_radius_m"] == hybrid_scanmapper.scanradius
    assert_start_zones_are_valid(hybrid_scanmapper, hybrid_result["drone_start_xy"], min_distance_m=1.2)
    assert_route_centerlines_do_not_overlap(hybrid_scanmapper, hybrid_result)
    assert_loop_lengths_are_balanced(hybrid_result)
    assert_swarm_result_is_valid(hybrid_scanmapper, hybrid_result, "Hybrid patrol routes")


def test_tunable_patrol_route_parameters():
    scanmapper = make_test_scanmapper()
    result = optimize_tunable_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=["H1", "H2", "H3"],
        search_radius_m=None,
        coverage_weight=0.65,
        obstacle_weight=0.70,
        target_coverage_ratio=0.80,
        candidate_spacing_m=0.4,
        max_patrol_points=30,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        obstacle_radius_m=1.2,
        min_point_score=1.0,
        path_separation_m=0.25,
        route_balance_weight=1.0,
        equalize_loop_lengths=True,
        loop_route=True,
        two_opt_passes=2,
        random_seed=23,
    )

    assert_planner_result_integrated(
        scanmapper,
        result,
        "tunable/custom",
        min_coverage_ratio=0.80,
        max_route_overlap_cells=0,
    )
    assert result["target_coverage_ratio"] == 0.80
    print("\nTunable patrol route:")
    print(
        f"  selected_points={len(result['selected_points'])} "
        f"coverage={result['coverage_ratio']:.1%}"
    )


def test_start_zone_generation_and_visualization():
    scanmapper = make_test_scanmapper()
    starts = generate_safe_drone_start_zones(
        scanmapper=scanmapper,
        drone_ids=["H1", "H2", "H3"],
        min_distance_m=1.5,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        random_seed=42,
    )
    assert_start_zones_are_valid(scanmapper, starts, min_distance_m=1.5)

    result = optimize_hybrid_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=["H1", "H2", "H3"],
        drone_start_xy=starts,
        search_radius_m=None,
        candidate_spacing_m=0.4,
        max_patrol_points=24,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        obstacle_radius_m=1.2,
    )

    output_dir = os.path.join(os.path.dirname(__file__), "dryrun_outputs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "swarm_patrol_search_areas.png")
    render_patrol_result_map(scanmapper, result, output_path)

    assert os.path.exists(output_path), "visualization image was not written"
    assert os.path.getsize(output_path) > 0, "visualization image is empty"
    print(f"\nVisualization written: {output_path}")


def test_close_physical_starts_do_not_control_patrol_sections():
    scanmapper = make_horizontal_wall_scanmapper()
    physical_starts = {
        "H1": (0.8, 0.8),
        "H2": (0.9, 0.8),
        "H3": (1.0, 0.8),
    }

    result = optimize_hybrid_swarm_patrol_routes(
        scanmapper=scanmapper,
        drone_ids=["H1", "H2", "H3"],
        drone_start_xy=physical_starts,
        search_radius_m=None,
        candidate_spacing_m=0.4,
        max_patrol_points=24,
        clearance_cells=DEFAULT_CLEARANCE_CELLS,
        buffer_cells=DEFAULT_BUFFER_CELLS,
        obstacle_radius_m=1.2,
        min_start_distance_m=1.2,
        random_seed=77,
    )

    patrol_starts = result["patrol_start_xy"]
    assert result["physical_start_xy"] == physical_starts
    assert patrol_starts != physical_starts
    assert_start_zones_are_valid(scanmapper, patrol_starts, min_distance_m=1.2)
    assert_planner_result_integrated(
        scanmapper,
        result,
        "close_physical_starts/hybrid",
        min_coverage_ratio=0.80,
        max_route_overlap_cells=0,
    )

    east_values = [point[0] for point in patrol_starts.values()]
    assert max(east_values) - min(east_values) >= 2.0, (
        "generated patrol starts are not spread across the arena"
    )

    for drone_id, route in result["routes"].items():
        assert route["waypoints"][0] == patrol_starts[drone_id]

    print("\nClose physical starts do not control patrol sections:")
    print(f"  physical_starts={physical_starts}")
    print(f"  patrol_starts={patrol_starts}")


def run_planner_for_scenario(scanmapper, planner_name, random_seed):
    common_kwargs = {
        "scanmapper": scanmapper,
        "drone_ids": ["H1", "H2", "H3"],
        "search_radius_m": None,
        "max_patrol_points": 24,
        "clearance_cells": DEFAULT_CLEARANCE_CELLS,
        "buffer_cells": DEFAULT_BUFFER_CELLS,
        "min_start_distance_m": 1.2,
        "random_seed": random_seed,
    }

    if planner_name == "frontage":
        return optimize_swarm_patrol_routes(
            candidate_spacing_m=0.4,
            **common_kwargs,
        )

    if planner_name == "obstacle":
        obstacle_kwargs = common_kwargs.copy()
        obstacle_kwargs["max_patrol_points"] = 33
        return optimize_obstacle_oriented_patrol_routes(
            candidate_spacing_m=0.4,
            obstacle_radius_m=1.2,
            **obstacle_kwargs,
        )

    if planner_name == "hybrid":
        return optimize_hybrid_swarm_patrol_routes(
            candidate_spacing_m=0.4,
            obstacle_radius_m=1.2,
            **common_kwargs,
        )

    raise ValueError(f"Unknown planner: {planner_name}")


def test_multiple_map_scenarios():
    scenarios = [
        ("open", make_open_scanmapper, 0.80),
        ("wall_corridors", make_test_scanmapper, 0.80),
        ("horizontal_walls", make_horizontal_wall_scanmapper, 0.80),
        ("block_clusters", make_block_cluster_scanmapper, 0.80),
        ("random_obstacles", make_random_obstacle_scanmapper, 0.80),
    ]
    planners = ["frontage", "obstacle", "hybrid"]

    output_dir = os.path.join(os.path.dirname(__file__), "dryrun_outputs", "robust_scenarios")
    os.makedirs(output_dir, exist_ok=True)

    print("\nRobust scenario sweep:")
    for scenario_index, (scenario_name, scenario_factory, min_coverage_ratio) in enumerate(scenarios):
        for planner_index, planner_name in enumerate(planners):
            scanmapper = scenario_factory()
            seed = 1000 + (scenario_index * 100) + planner_index
            result = run_planner_for_scenario(scanmapper, planner_name, random_seed=seed)
            label = f"{scenario_name}/{planner_name}"

            assert_planner_result_integrated(
                scanmapper,
                result,
                label,
                min_coverage_ratio=min_coverage_ratio,
                max_route_overlap_cells=0,
            )

            output_path = os.path.join(output_dir, f"{scenario_name}_{planner_name}.png")
            render_patrol_result_map(scanmapper, result, output_path)
            assert os.path.exists(output_path), f"{label} visualization was not written"
            assert os.path.getsize(output_path) > 0, f"{label} visualization is empty"

            route_lengths = {
                drone_id: round(route["length_m"], 2)
                for drone_id, route in result["routes"].items()
            }
            print(
                f"  {label}: "
                f"coverage={result['coverage_ratio']:.1%} "
                f"scan_overlap={result['overlap_cells']} "
                f"route_overlap={count_route_centerline_overlap(scanmapper, result)} "
                f"starts={result['drone_start_xy']} "
                f"lengths={route_lengths}"
            )


if __name__ == "__main__":
    test_astar_route_on_cluttered_map()
    test_optimized_swarm_patrol_routes()
    test_obstacle_and_hybrid_patrol_routes()
    test_tunable_patrol_route_parameters()
    test_start_zone_generation_and_visualization()
    test_close_physical_starts_do_not_control_patrol_sections()
    test_multiple_map_scenarios()
