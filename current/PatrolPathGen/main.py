# main.py (Part 1 of 3)
import os
import numpy as np
import copy  
import napari
from magicgui import magicgui
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QTimer

from dilation_engine import generate_dilation_map
from clustering_engine import compute_workload_regions
from path_engine import generate_drone_path, smooth_final_trajectory

# ----------------------------------------------------
# TOP-LEVEL INDEPENDENT DRONE TAKEOFF COORDINATES
# ----------------------------------------------------
drone_last_positions = {
    1: (5.0, 5.0),    # Drone 1 Start (Teal)
    2: (50.0, 10.0),  # Drone 2 Start (Magenta)
    3: (95.0, 45.0)   # Drone 3 Start (Khaki)
}

ARENA_HEIGHT = 110
ARENA_WIDTH = 55

# Global runtime variables tracking data structures
obstacle_data = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
frequency_data = np.ones((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
active_targets = []
graph_network_cached = []

# Instantiate the UI window container
viewer = napari.Viewer()

# ----------------------------------------------------
# COLLISION-AWARE KINODYNAMIC ENGINES
# ----------------------------------------------------
def calculate_distance(p1, p2):
    """Calculates Euclidean spatial distance between two points."""
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_angle_penalty(theta):
    """Bakes kinodynamic braking weights directly into virtual extra meters."""
    if 0.0 <= theta <= 30.0:
        return 0.0
    elif 30.0 < theta <= 60.0:
        return 2.0
    elif 60.0 < theta <= 90.0:
        return 7.0
    elif 90.0 < theta <= 135.0:
        return 15.0
    return 30.0

def calculate_vertex_angle(p_prev, p_curr, p_next):
    """Computes the interior angle deviation (theta) at Node B."""
    v1 = np.array(p_curr) - np.array(p_prev)
    v2 = np.array(p_next) - np.array(p_curr)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    cos_theta = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def evaluate_total_cost(route, points):
    """Dynamic Evaluator: Accumulates Spatial Distance + Angle Penalties."""
    if len(route) < 2:
        return 0.0
    total_cost = 0.0
    for i in range(len(route) - 1):
        total_cost += calculate_distance(points[route[i]], points[route[i+1]])
    for i in range(1, len(route) - 1):
        theta = calculate_vertex_angle(points[route[i-1]], points[route[i]], points[route[i+1]])
        total_cost += get_angle_penalty(theta)
    return total_cost

def solve_kinodynamic_tsp_2opt(points, max_iterations=200):
    """Reshuffling engine minimizing total virtual flight path length."""
    num_points = len(points)
    if num_points < 3:
        return list(range(num_points)), 0.0
    best_route = list(range(num_points))
    best_cost = evaluate_total_cost(best_route, points)
    improved = True
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        for i in range(1, num_points - 1):  # Keeps starting launch index 0 locked
            for j in range(i + 1, num_points):
                new_route = copy.deepcopy(best_route)
                new_route[i:j+1] = reversed(new_route[i:j+1])
                new_cost = evaluate_total_cost(new_route, points)
                if new_cost < best_cost:
                    best_route = new_route
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return best_route, best_cost

def optimize_path_coordinates(raw_path_coords):
    """Helper wrapper to transform raw path engine steps into optimized sequences."""
    if raw_path_coords is None or len(raw_path_coords) < 2:
        return raw_path_coords
    best_sequence, _ = solve_kinodynamic_tsp_2opt(raw_path_coords)
    return raw_path_coords[best_sequence]

def enforce_obstacle_clearance(path_coords, obs_map):
    """
    Scans every coordinate on the continuous trajectory vector against the 
    Obstacle Grid Map. If an element hits a wall, it steps back out into the open.
    """
    if path_coords is None or len(path_coords) == 0:
        return path_coords
        
    h_max = obs_map.shape[0] - 1
    w_max = obs_map.shape[1] - 1
    validated_path = path_coords.copy()
    
    for i in range(len(validated_path)):
        y, x = validated_path[i]
        cy, cx = int(np.clip(y, 0, h_max)), int(np.clip(x, 0, w_max))
        
        if obs_map[cy, cx] == 1:
            found_escape = False
            for r in range(1, 6):
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny <= h_max and 0 <= nx <= w_max:
                            if obs_map[ny, nx] == 0:
                                validated_path[i] = [float(ny), float(nx)]
                                found_escape = True
                                break
                    if found_escape:
                        break
                if found_escape:
                    break
                    
    return validated_path

# main.py (Part 2 of 3)

# ----------------------------------------------------
# PROGRAMMATIC API EXTERNAL INTERFACES
# ----------------------------------------------------
def init(initial_array=None):
    """
    API Function 1: Boots up the Napari editor environment.
    Accepts an initial array where 0 = obstacle present, 1 = open space.
    If no parameter is passed, instantiates a clean, open arena layout.
    """
    global obstacle_data, frequency_data
    
    if initial_array is not None:
        print("Streamlined API: Injected initial configuration layout matrix.")
        # Flip binary formatting: convert friend's 0 (obstacle) into our 1 (obstacle logic)
        obstacle_data = np.where(initial_array == 0, 1, 0).astype(np.uint8)
    else:
        print("Streamlined API: No array provided. Generating clean open arena...")
        obstacle_data = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
        
    # Inject secure structural bounding perimeter frames
    obstacle_data[0, :], obstacle_data[-1, :], obstacle_data[:, 0], obstacle_data[:, -1] = 1, 1, 1, 1
    
    # Initialize basic grid space mapping automatically
    frequency_data = np.ones((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
    
    # Render baseline datasets straight onto the viewer layers stack
    obs_layer.data = obstacle_data
    dilation_layer.data = np.zeros_like(obstacle_data)
    freq_layer.data = frequency_data
    region_layer.data = np.zeros_like(obstacle_data)
    
    print("UI Notification: Editor live. Proceed through the visual sidebar workflow channels.")

def waypoints(total_waypoints=15, alpha_high=1.1, scale_high=1.8, alpha_low=1.99, scale_low=25.0):
    """
    API Function 2: Calculates paths, minimizes angles, smoothes trajectories, 
    evades obstacles, updates history cache, and RETURNS a list of 3 drone coordinate arrays.
    """
    global drone_last_positions
    reg_map = region_layer.data
    freq_map = freq_layer.data
    
    if np.max(reg_map) == 0:
        print("Error: Search regions are empty! Execute Step 3 in the UI first."); return []
        
    # Calculate flight tracks using localized position continuity logs
    raw_p1 = generate_drone_path(reg_map, freq_map, region_id=1, num_waypoints=total_waypoints, 
                             alpha_high=alpha_high, scale_high=scale_high, alpha_low=alpha_low, scale_low=scale_low,
                             start_pos=drone_last_positions[1])
    raw_p2 = generate_drone_path(reg_map, freq_map, region_id=2, num_waypoints=total_waypoints,
                             alpha_high=alpha_high, scale_high=scale_high, alpha_low=alpha_low, scale_low=scale_low,
                             start_pos=drone_last_positions[2])
    raw_p3 = generate_drone_path(reg_map, freq_map, region_id=3, num_waypoints=total_waypoints,
                             alpha_high=alpha_high, scale_high=scale_high, alpha_low=alpha_low, scale_low=scale_low,
                             start_pos=drone_last_positions[3])
                             
    # Permute index sequences to save kinetic battery energy
    p1_ordered = optimize_path_coordinates(raw_p1)
    p2_ordered = optimize_path_coordinates(raw_p2)
    p3_ordered = optimize_path_coordinates(raw_p3)
    
    # Smooth jagged flight tracks into clean curves
    p1_smooth = smooth_final_trajectory(p1_ordered, ARENA_HEIGHT-1, ARENA_WIDTH-1)
    p2_smooth = smooth_final_trajectory(p2_ordered, ARENA_HEIGHT-1, ARENA_WIDTH-1)
    p3_smooth = smooth_final_trajectory(p3_ordered, ARENA_HEIGHT-1, ARENA_WIDTH-1)
    
    # Correct positions to push drone tracks clear of walls
    p1 = enforce_obstacle_clearance(p1_smooth, dilation_layer.data)
    p2 = enforce_obstacle_clearance(p2_smooth, dilation_layer.data)
    p3 = enforce_obstacle_clearance(p3_smooth, dilation_layer.data)
    
    # Log newest terminal nodes as the start coordinates for the next cycle
    drone_last_positions[1] = tuple(p1[-1])
    drone_last_positions[2] = tuple(p2[-1])
    drone_last_positions[3] = tuple(p3[-1])
    
    np.save('drone1_path.npy', p1); np.save('drone2_path.npy', p2); np.save('drone3_path.npy', p3)
    
    # Draw curves on the Napari dashboard interface
    path_render_layer.data = [p1, p2, p3]
    path_render_layer.edge_color = ['teal', 'magenta', 'khaki']
    path_render_layer.shape_type = ['path', 'path', 'path']
    
    # Return the 3 distinct sub-arrays exactly as requested
    return [p1, p2, p3]

# ----------------------------------------------------
# DISPLAY VISUAL LAYERS COMPILATION
# ----------------------------------------------------
dummy_grid = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)

obs_layer = viewer.add_labels(dummy_grid, name='1. Obstacle Map', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'transparent', 1:'darkgray'}))
dilation_layer = viewer.add_labels(dummy_grid, name='2. Dilated Flight Buffer', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'transparent', 1:(0.86, 0.08, 0.24, 0.3)}))
graph_layer = viewer.add_shapes(name='3. Target Graph Networks', shape_type='path', edge_color='cyan', edge_width=0.6, face_color='transparent')
freq_layer = viewer.add_labels(dummy_grid, name='4. Frequency Map', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'black', 1:'mediumpurple', 2:'orange', 3:'crimson'}))
region_layer = viewer.add_labels(dummy_grid, name='5. Drone Search Regions', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'transparent', 1:(0.12, 0.53, 0.53, 0.4), 2:(0.50, 0.00, 0.50, 0.4), 3:(0.93, 0.86, 0.51, 0.4)}))
path_render_layer = viewer.add_shapes(name='6. Smooth Drone Flight Paths', shape_type='path', edge_width=0.4, face_color='transparent')
target_render_layer = viewer.add_points(None, name='7. Active Convoys', size=3, face_color='yellow', border_color='orange')

# main.py (Part 3 of 3)

# ----------------------------------------------------
# MOUSE CANVAS CLICKS INTERCEPTORS
# ----------------------------------------------------
is_r_pressed = False
drag_start_coord = None

@viewer.bind_key('r', overwrite=True)
def hold_r_key(viewer):
    global is_r_pressed
    is_r_pressed = True
    yield
    is_r_pressed = False

@obs_layer.mouse_drag_callbacks.append
def rectangle_draw_callback(layer, event):
    global drag_start_coord, is_r_pressed
    if not is_r_pressed: return
    if event.type == 'mouse_press':
        drag_start_coord = event.position  
    elif event.type == 'mouse_release':
        if drag_start_coord is None: return
        drag_end_coord = event.position
        y_min, y_max = int(max(0, min(drag_start_coord, drag_end_coord))), int(min(ARENA_HEIGHT, max(drag_start_coord, drag_end_coord)))
        x_min, x_max = int(max(0, min(drag_start_coord, drag_end_coord))), int(min(ARENA_WIDTH, max(drag_start_coord, drag_end_coord)))
        current_data = layer.data.copy()
        current_data[y_min:y_max+1, x_min:x_max+1] = 1
        layer.data = current_data
        drag_start_coord = None

# ----------------------------------------------------
# STREAMLINED SIDEBAR WORKFLOW WIDGET PANELS
# ----------------------------------------------------
@magicgui(call_button="Step 1: Compile Dilation Layer", mode={"choices": ["radial", "square"]}, radius={"min": 1, "max": 15})
def dilation_widget(mode='radial', radius=3):
    dilation_layer.data = generate_dilation_map(obs_layer.data, mode=mode, radius=radius)
    freq_layer.data = np.where(dilation_layer.data == 1, 0, freq_layer.data.copy())
    print("Engine Workflow: Step 1 Complete. Obstacle expansion mapped to flight buffer layers.")

@magicgui(call_button="Step 2: Save Environment & Run Convoys", num_convoys={"min": 1, "max": 10})
def simulation_widget(num_convoys=3):
    global active_targets, graph_network_cached
    sanitized_frequency = np.where(obs_layer.data == 1, 0, freq_layer.data.copy())
    np.save('arena_obstacles.npy', obs_layer.data)
    np.save('frequency_map.npy', sanitized_frequency)
    
    graph_network_cached = graph_layer.data 
    active_targets.clear()
    target_render_layer.data = np.empty((0, 2))
        
    if len(graph_network_cached) == 0: return
    for _ in range(num_convoys):
        convoy = TargetConvoy(current_segment_idx=np.random.randint(0, len(graph_network_cached)), direction=np.random.choice([1, -1]))
        convoy.progress = np.random.rand() 
        active_targets.append(convoy)
    print("Engine Workflow: Step 2 Complete. Multi-target active simulation loops loaded.")

@magicgui(call_button="Step 3: Compute Drone Search Regions")
def region_widget():
    region_layer.data = compute_workload_regions(freq_layer.data)
    np.save('drone_regions.npy', region_layer.data)
    print("Engine Workflow: Step 3 Complete. Continuous workload regions calculated via Geodesic K-Medoids.")

@magicgui(
    call_button="Step 4: Generate Drone Flight Paths",
    total_waypoints={"min": 5, "max": 500, "step": 5},
    alpha_high={"min": 1.01, "max": 1.99, "step": 0.05, "label": "Alpha (High)"},
    scale_high={"min": 0.5, "max": 50.0, "step": 0.5, "label": "Scale (High)"},
    alpha_low={"min": 1.01, "max": 1.99, "step": 0.01, "label": "Alpha (Low)"},
    scale_low={"min": 0.5, "max": 50.0, "step": 0.5, "label": "Scale (Low)"}
)
def waypoints_widget(total_waypoints=15, alpha_high=1.1, scale_high=1.8, alpha_low=1.99, scale_low=25.0):
    # Triggers background API solver, updates layers, and returns coordinate arrays
    waypoints(total_waypoints, alpha_high, scale_high, alpha_low, scale_low)

viewer.window.add_dock_widget(dilation_widget, area='right', name="Workflow Module 1")
viewer.window.add_dock_widget(simulation_widget, area='right', name="Workflow Module 2")
viewer.window.add_dock_widget(region_widget, area='right', name="Workflow Module 3")
viewer.window.add_dock_widget(waypoints_widget, area='right', name="Workflow Module 4")

# ----------------------------------------------------
# REALTIME INTERPOLATION TICK TICK LOOPS
# ----------------------------------------------------
def simulation_step():
    global active_targets, graph_network_cached
    if len(active_targets) == 0 or len(graph_network_cached) == 0: return
    render_coordinates = []
    
    for convoy in active_targets:
        if convoy.segment_idx >= len(graph_network_cached): continue
        coords = graph_network_cached[convoy.segment_idx]
        convoy.progress += convoy.speed * convoy.direction
        
        if convoy.progress >= 1.0 or convoy.progress <= 0.0:
            junction_pt = coords[-1, :] if convoy.progress >= 1.0 else coords[0, :]
            valid_branches = []
            for idx, net_path in enumerate(graph_network_cached):
                if net_path.ndim < 2: continue
                if np.linalg.norm(net_path[0, :] - junction_pt) < 2.5: valid_branches.append((idx, 1))
                if np.linalg.norm(net_path[-1, :] - junction_pt) < 2.5: valid_branches.append((idx, -1))
            if len(valid_branches) > 0:
                next_seg, next_dir = valid_branches[np.random.randint(0, len(valid_branches))]
                convoy.segment_idx = next_seg; convoy.direction = next_dir
                convoy.progress = 0.0 if next_dir == 1 else 1.0
            else:
                convoy.direction *= -1; convoy.progress = np.clip(convoy.progress, 0.0, 1.0)
                
        current_coords = graph_network_cached[convoy.segment_idx]
        num_segments = len(current_coords) - 1
        if num_segments <= 0:
            render_coordinates.append(current_coords[0, :] if current_coords.ndim > 1 else current_coords); continue
            
        scaled_prog = convoy.progress * num_segments
        segment_num = np.clip(int(np.floor(scaled_prog)), 0, num_segments - 1)
        local_t = scaled_prog - segment_num
        p0 = current_coords[segment_num, :] if current_coords.ndim > 1 else current_coords
        p1 = current_coords[segment_num + 1, :] if current_coords.ndim > 1 else current_coords
        
        pt = p0 + local_t * (p1 - p0)
        render_coordinates.append(pt)
        
    if render_coordinates:
        target_render_layer.data = np.array(render_coordinates)

timer = QTimer(); timer.timeout.connect(simulation_step); timer.start(50)

# ----------------------------------------------------
# LAUNCH BASELINE ARENA BOOTSTRAP
# ----------------------------------------------------
if __name__ == '__main__':
    # Launches an empty template map out-of-the-box on direct console executions
    init(None)
    napari.run()
