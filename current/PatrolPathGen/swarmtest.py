import numpy as np
import napari
from ScanMap import ScanMapper
from Astar import simplifypath, pathfind
from optimized_swarm_path_planner import (
    DEFAULT_BUFFER_CELLS,
    DEFAULT_CLEARANCE_CELLS,
    optimize_hybrid_swarm_patrol_routes,
    optimize_obstacle_oriented_patrol_routes,
    optimize_swarm_patrol_routes,
    optimize_tunable_swarm_patrol_routes,
)

SEARCHLIGHTRADIUS_m = 0.7
DEFAULT_HULA_START_XY = {
    "H1": (0.8, 0.8),
    "H2": (0.8, 1.8),
    "H3": (0.8, 2.8),
}
DEFAULT_HULA_DRONE_IDS = list(DEFAULT_HULA_START_XY.keys())

scanmapper = ScanMapper(
    110, 55, 0.1, 0, 0, None, SEARCHLIGHTRADIUS_m
)

scanmapper.scanmap[scanmapper.scanmap==2] = 1
my_array = scanmapper.scanmap.copy()
viewer = napari.Viewer()
labels_layer = viewer.add_labels(my_array, name='My Paint Layer')


# - SET FREE TILES TO 0
# - SET OBSTACLE TILES TO 1
napari.run() # Execution pauses here until you close the napari window
# 1. Grab data and enforce standard 32-bit integers (avoids data-type errors)
edited_array = labels_layer.data.astype(np.int32)

# 2. Instantiate ScanMapper cleanly with a fresh canvas setup
scanmapper = ScanMapper(
    110, 55, 0.1, 0, 0, None, SEARCHLIGHTRADIUS_m
)

# 3. Explicitly overwrite the scanmap grid variable inside the class instance
scanmapper.scanmap = edited_array


# the gridmap is scanmapper.scanmap
# The gridmap is a 2D array of ints and represents an 11m x 5.5m arena. To "scan" an area, call
# scanmapper.searchlight(DRONE_X_COORD_IN_METRES, DRONE_Y_COORD_IN_METRES)
# - it will automatically calculate which cells in the 110 x 55 gridmap are within SEARCHLIGHTRADIUS_m of the drone,
# - and set them to 8 (searched)

# # TODO YOUR CODE HERE: ALGO FOR DRONE TO GENERATE A PATROL PATH BASED ON scanmapper.scanmap
# - scanmapper.scanmap is a 2D array of ints, 
# # SCANMAP LEGEND
# # 1 - free
# # 2 - obstacle
# # 8 - searchlight scanned this cell

def generate_patrol_path(mode, drone_start_xy=None, return_metadata=False, **planner_kwargs):
    '''Generates a patrol path by giving a list of waypoints for each of the 3 drones to follow. 
    Each drone's waypoints are ordered in the sequence they are to be followed.
    - the drone will loop these waypoints. '''
    if mode == 1:
        return obstacle_oriented_pathfind(
            drone_start_xy=drone_start_xy,
            return_metadata=return_metadata,
            **planner_kwargs,
        )
    elif mode == 2:
        return frontage_oriented_pathfind(
            drone_start_xy=drone_start_xy,
            return_metadata=return_metadata,
            **planner_kwargs,
        )
    elif mode == 3:
        return hybrid_pathfind(
            drone_start_xy=drone_start_xy,
            return_metadata=return_metadata,
            **planner_kwargs,
        )
    elif mode in (4, "custom", "tunable"):
        return tunable_pathfind(
            drone_start_xy=drone_start_xy,
            return_metadata=return_metadata,
            **planner_kwargs,
        )
    else:
        raise ValueError("Invalid mode. Choose 1=obstacle, 2=frontage, 3=hybrid, or 4=tunable.")

def _format_planner_result(result, return_metadata):
    if return_metadata:
        return result

    return {
        drone_id: route["waypoints"]
        for drone_id, route in result["routes"].items()
    }


def _ensure_planner_scanmap_legend(scanmapper_instance):
    """Planner expects 1=free, 2=obstacle, 8=searched."""
    unique_values = set(np.unique(scanmapper_instance.scanmap).tolist())

    if 0 in unique_values and 2 not in unique_values:
        normalized = scanmapper_instance.scanmap.copy()
        normalized[scanmapper_instance.scanmap == 0] = 1
        normalized[scanmapper_instance.scanmap == 1] = 2
        scanmapper_instance.scanmap = normalized.astype(np.int32)

    return scanmapper_instance


def obstacle_oriented_pathfind(
    scanmapper_instance=None,
    drone_start_xy=None,
    return_metadata=False,
    search_radius_m=None,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=33,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    obstacle_radius_m=1.2,
): 
    '''Based on the idea that obstacle determines the pathing of the drone.
        1. Determines 3 areas of high obstacle density, (able to be tuned by a parameter)
        2. Each drone is to encircle an area of density
        3. Large areas with little density are 'added' on to existing patrol points, by making a figure 8 pattern or such
            - these areas of light density are given priority, maybe 2 loops of lighter density and 1 loop of heavier density, or something like that'''
    active_scanmapper = _ensure_planner_scanmap_legend(scanmapper_instance or scanmapper)

    result = optimize_obstacle_oriented_patrol_routes(
        scanmapper=active_scanmapper,
        drone_ids=DEFAULT_HULA_DRONE_IDS,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
        obstacle_radius_m=obstacle_radius_m,
    )

    return _format_planner_result(result, return_metadata)


def tunable_pathfind(
    scanmapper_instance=None,
    drone_start_xy=None,
    return_metadata=False,
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
    path_separation_m=0.25,
    route_balance_weight=1.0,
    equalize_loop_lengths=True,
    loop_route=True,
    two_opt_passes=2,
):
    """Single tunable patrol planner used by the preset modes."""
    active_scanmapper = _ensure_planner_scanmap_legend(scanmapper_instance or scanmapper)

    result = optimize_tunable_swarm_patrol_routes(
        scanmapper=active_scanmapper,
        drone_ids=DEFAULT_HULA_DRONE_IDS,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        coverage_weight=coverage_weight,
        obstacle_weight=obstacle_weight,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
        obstacle_radius_m=obstacle_radius_m,
        min_point_score=min_point_score,
        min_start_distance_m=min_start_distance_m,
        path_separation_m=path_separation_m,
        route_balance_weight=route_balance_weight,
        equalize_loop_lengths=equalize_loop_lengths,
        loop_route=loop_route,
        two_opt_passes=two_opt_passes,
    )

    return _format_planner_result(result, return_metadata)


def frontage_oriented_pathfind(
    scanmapper_instance=None,
    drone_start_xy=None,
    return_metadata=False,
    search_radius_m=None,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=24,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
): 
    '''Based on the idea that frontage determines the pathing of the drone.
        1. Identifies areas with high camera coverage/frontage
        2. Greedily select the best patrol points.
        3. Split selected patrol points evenly across drones.
        4. Order each drone's points with nearest-neighbor route planning.
        5. Improve route order with 2-opt.
        6. Connect ordered points using A*.
        '''
    active_scanmapper = _ensure_planner_scanmap_legend(scanmapper_instance or scanmapper)

    result = optimize_swarm_patrol_routes(
        scanmapper=active_scanmapper,
        drone_ids=DEFAULT_HULA_DRONE_IDS,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
    )

    return _format_planner_result(result, return_metadata)


def hybrid_pathfind(
    scanmapper_instance=None,
    drone_start_xy=None,
    return_metadata=False,
    search_radius_m=None,
    target_coverage_ratio=None,
    candidate_spacing_m=0.4,
    max_patrol_points=24,
    clearance_cells=DEFAULT_CLEARANCE_CELLS,
    buffer_cells=DEFAULT_BUFFER_CELLS,
    obstacle_radius_m=1.2,
): 
    '''Balanced patrol strategy.
        1. Still maximises camera/search frontage over free space.
        2. Gives extra score to points near obstacle edges and corridors.
        3. Splits patrol points evenly across the 3 HULA drones.
        4. Orders each route and connects it safely using A*.'''
    active_scanmapper = _ensure_planner_scanmap_legend(scanmapper_instance or scanmapper)

    result = optimize_hybrid_swarm_patrol_routes(
        scanmapper=active_scanmapper,
        drone_ids=DEFAULT_HULA_DRONE_IDS,
        drone_start_xy=drone_start_xy,
        search_radius_m=search_radius_m,
        target_coverage_ratio=target_coverage_ratio,
        candidate_spacing_m=candidate_spacing_m,
        max_patrol_points=max_patrol_points,
        clearance_cells=clearance_cells,
        buffer_cells=buffer_cells,
        obstacle_radius_m=obstacle_radius_m,
    )

    return _format_planner_result(result, return_metadata)
