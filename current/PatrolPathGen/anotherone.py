# main.py
import os
import numpy as np
import napari
from magicgui import magicgui
from napari.utils.colormaps import DirectLabelColormap
from qtpy.QtCore import QTimer

from dilation_engine import generate_dilation_map
from clustering_engine import compute_workload_regions
from path_engine import generate_drone_path

ARENA_HEIGHT = 55
ARENA_WIDTH = 110

obstacle_data = np.zeros((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)
obstacle_data[0, :], obstacle_data[-1, :], obstacle_data[:, 0], obstacle_data[:, -1] = 1, 1, 1, 1
frequency_data = np.ones((ARENA_HEIGHT, ARENA_WIDTH), dtype=np.uint8)

class TargetConvoy:
    def __init__(self, current_segment_idx, direction=1):
        self.segment_idx = current_segment_idx  
        self.progress = 0.0                     
        self.speed = 0.05                       
        self.direction = direction              

active_targets = []
graph_network_cached = []

viewer = napari.Viewer()

# ----------------------------------------------------
# LAYER STACK CONFIGURATION
# ----------------------------------------------------
obs_layer = viewer.add_labels(obstacle_data, name='1. Obstacle Map', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'transparent', 1:'darkgray'}))
dilation_layer = viewer.add_labels(np.zeros_like(obstacle_data), name='2. Dilated Flight Buffer', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'transparent', 1:(0.86, 0.08, 0.24, 0.3)}))
graph_layer = viewer.add_shapes(name='3. Target Graph Networks', shape_type='path', edge_color='cyan', edge_width=0.6, face_color='transparent')
freq_layer = viewer.add_labels(frequency_data, name='4. Frequency Map', colormap=DirectLabelColormap(color_dict={None:'transparent', 0:'black', 1:'mediumpurple', 2:'orange', 3:'crimson'}))

region_layer = viewer.add_labels(np.zeros_like(obstacle_data), name='5. Drone Search Regions', colormap=DirectLabelColormap(color_dict={
    None:'transparent', 0:'transparent', 1:(0.12, 0.53, 0.53, 0.4), 2:(0.50, 0.00, 0.50, 0.4), 3:(0.93, 0.86, 0.51, 0.4)
}))

path_render_layer = viewer.add_shapes(name='6. Smooth Drone Flight Paths', shape_type='path', edge_width=0.4, face_color='transparent')
target_render_layer = viewer.add_points(None, name='7. Active Convoys', size=3, face_color='yellow', border_color='orange')

# ----------------------------------------------------
# MOUSE CANVAS OBS OVERRIDES
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
# UI STEP CONTROL INTERFACES
# ----------------------------------------------------
@magicgui(call_button="1. Compile Dilation Layer", mode={"choices": ["radial", "square"]}, radius={"min": 1, "max": 15})
def dilation_widget(mode='radial', radius=3):
    dilation_layer.data = generate_dilation_map(obs_layer.data, mode=mode, radius=radius)
    freq_layer.data = np.where(dilation_layer.data == 1, 0, freq_layer.data.copy())
    print("Engine Action: Step 2 Complete. Boundaries copied into Frequency Grid Map.")

@magicgui(call_button="2. Save Matrices & Run Convoys", num_convoys={"min": 1, "max": 10})
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
    print("Engine Action: Configuration saved. Multi-target simulation live.")

@magicgui(call_button="3. Compute Drone Regions (Geodesic)")
def region_widget():
    region_layer.data = compute_workload_regions(freq_layer.data)
    np.save('drone_regions.npy', region_layer.data)
    print("Success: Continuous workload regions calculated via Geodesic K-Medoids.")

# FIXED UPDATED PATH_WIDGET: Exposes all internal alpha/scale variables directly to sliders
@magicgui(
    call_button="4. Generate Smooth Drone Paths (Lévy)",
    total_waypoints={"min": 10, "max": 500, "step": 5},
    alpha_high={"min": 1.01, "max": 1.99, "step": 0.05, "label": "Alpha (High Freq)"},
    scale_high={"min": 0.5, "max": 15.0, "step": 0.5, "label": "Scale (High Freq)"},
    alpha_low={"min": 1.01, "max": 1.99, "step": 0.05, "label": "Alpha (Low Freq)"},
    scale_low={"min": 0.5, "max": 25.0, "step": 0.5, "label": "Scale (Low Freq)"}
)
def path_widget(total_waypoints=15, alpha_high=1.1, scale_high=1.8, alpha_low=1.9, scale_low=25):
    reg_map = region_layer.data
    freq_map = freq_layer.data
    
    if np.max(reg_map) == 0:
        print("Warning: Please compute step 3 regions before generating paths!"); return
        
    print(f"Generating {total_waypoints}-step smooth tuned Levy flights for all 3 drones...")
    p1 = generate_drone_path(reg_map, freq_map, region_id=1, num_waypoints=total_waypoints, 
                             alpha_high=alpha_high, scale_high=scale_high, alpha_low=alpha_low, scale_low=scale_low)
    p2 = generate_drone_path(reg_map, freq_map, region_id=2, num_waypoints=total_waypoints,
                             alpha_high=alpha_high, scale_high=scale_high, alpha_low=alpha_low, scale_low=scale_low)
    p3 = generate_drone_path(reg_map, freq_map, region_id=3, num_waypoints=total_waypoints,
                             alpha_high=alpha_high, scale_high=scale_high, alpha_low=alpha_low, scale_low=scale_low)
    
    np.save('drone1_path.npy', p1); np.save('drone2_path.npy', p2); np.save('drone3_path.npy', p3)
    
    path_render_layer.data = []
    path_render_layer.add(p1, shape_type='path', edge_color='teal')
    path_render_layer.add(p2, shape_type='path', edge_color='magenta')
    path_render_layer.add(p3, shape_type='path', edge_color='khaki')
    print("Success: Smooth custom search paths compiled and written to disk files.")

viewer.window.add_dock_widget(dilation_widget, area='right', name="Step 2 Engine")
viewer.window.add_dock_widget(simulation_widget, area='right', name="Step 3 Init")
viewer.window.add_dock_widget(region_widget, area='right', name="Step 3 Slicing")
viewer.window.add_dock_widget(path_widget, area='right', name="Step 4 Navigation")

# ----------------------------------------------------
# REALTIME MEMORY-SAFE TICK INTERPOLATION ENGINE
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

if __name__ == '__main__':
    napari.run()
