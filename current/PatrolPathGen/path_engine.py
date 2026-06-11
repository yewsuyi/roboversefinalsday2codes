# path_engine.py
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.special import gamma  # Fixed modern numpy compatibility fix

def generate_levy_step(alpha, scale=1.0):
    """Generates a bounded step length from a stable heavy-tailed Levy distribution using Mantegna's algorithm."""
    # Replaced np.math.gamma with scipy.special.gamma to prevent runtime errors
    sigma_num = (gamma(1 + alpha) * np.sin(np.pi * alpha / 2))
    sigma_den = (gamma((1 + alpha) / 2) * alpha * (2**((alpha - 1) / 2)))
    sigma_u = (sigma_num / sigma_den) ** (1 / alpha)
    
    u = np.random.normal(0, sigma_u)
    v = np.random.normal(0, 1)
    step_length = u / (np.abs(v) ** (1 / alpha))
    return np.clip(step_length * scale, 2.0, 25.0)
# path_engine.py (Update this function block)

def generate_drone_path(region_matrix, frequency_matrix, region_id, num_waypoints=30, 
                        alpha_high=1.1, scale_high=1.8, alpha_low=1.9, scale_low=7.5,
                        start_pos=None):  # <--- Added start_pos parameter
    """
    Step 4: Generates a perfectly region-bounded adaptive Levy Flight track.
    Accepts a start_pos to maintain flawless positional continuity across generational updates.
    """
    region_mask = (region_matrix == region_id)
    y_bounds, x_bounds = np.where(region_mask)
    if len(y_bounds) == 0:
        return np.empty((0, 2))
        
    # Isolate mandatory 'Must Visit' (Value 3) pixels inside this region
    must_visit_y, must_visit_x = np.where((frequency_matrix == 3) & region_mask)
    must_visit_points = np.column_stack((must_visit_y, must_visit_x))
    
    # CONTINUITY CHECK: If a persistent last waypoint exists, use it! Otherwise, fallback to base
    if start_pos is not None:
        current_pos = np.array(start_pos, dtype=float)
    else:
        current_pos = np.array([np.mean(y_bounds), np.mean(x_bounds)], dtype=float)
        
    raw_waypoints = [current_pos.copy()]
    
    target_raw_count = max(40, num_waypoints * 2)
    h_max, w_max = region_matrix.shape[0] - 1, region_matrix.shape[1] - 1
    
    for step in range(target_raw_count):
        cy = int(np.clip(current_pos[0], 0, h_max))
        cx = int(np.clip(current_pos[1], 0, w_max))
        
        # 3x3 Look-Ahead Horizon sampling
        y_slice = slice(max(0, cy-1), min(h_max+1, cy+2))
        x_slice = slice(max(0, cx-1), min(w_max+1, cx+2))
        local_window = frequency_matrix[y_slice, x_slice]
        
        if np.any(local_window >= 2):
            alpha, scale = alpha_high, scale_high  
        else:
            alpha, scale = alpha_low, scale_low    
            
        step_success = False
        for attempt in range(15):
            step_len = generate_levy_step(alpha, scale)
            angle = np.random.uniform(0, 2 * np.pi)
            proposed_vector = np.array([step_len * np.sin(angle), step_len * np.cos(angle)])
            
            steps_to_check = int(max(5, step_len))
            hit_wall = False
            valid_pos = current_pos.copy()
            
            for t_step in range(1, steps_to_check + 1):
                fraction = t_step / steps_to_check
                check_pos = current_pos + fraction * proposed_vector
                chk_y, chk_x = int(np.clip(check_pos[0], 0, h_max)), int(np.clip(check_pos[1], 0, w_max))
                
                if region_matrix[chk_y, chk_x] == region_id:
                    valid_pos = check_pos
                else:
                    hit_wall = True
                    break
            
            if not hit_wall or np.linalg.norm(valid_pos - current_pos) > 1.5:
                current_pos = valid_pos
                step_success = True
                break
        
        if not step_success:
            center_direction = np.array([np.mean(y_bounds), np.mean(x_bounds)]) - current_pos
            if np.linalg.norm(center_direction) > 0:
                current_pos += (center_direction / np.linalg.norm(center_direction)) * 3.0
                
        current_pos = np.clip(current_pos, 1, [h_max - 1, w_max - 1])
        if np.linalg.norm(current_pos - raw_waypoints[-1]) > 0.5:
            raw_waypoints.append(current_pos.copy())
        
    pts = np.array(raw_waypoints)
    indices = np.linspace(0, len(pts)-1, num_waypoints).astype(int)
    downsampled_path = pts[indices]
    
    if len(must_visit_points) > 0:
        for target_node in must_visit_points:
            distances = np.linalg.norm(downsampled_path - target_node, axis=1)
            closest_idx = np.argmin(distances)
            downsampled_path[closest_idx] = target_node.astype(float)
            
    return downsampled_path

# ----------------------------------------------------
# STANDALONE TRAJECTORY RESAMPLING & SMOOTHING ENGINE
# ----------------------------------------------------
def smooth_final_trajectory(optimized_points, h_max, w_max):
    """
    Applies B-Spline interpolation AFTER points have been ordered by the TSP engine.
    This creates an aerodynamically sound path for physical drone execution.
    """
    if len(optimized_points) < 3:
        return optimized_points
        
    try:
        # Upsample resolution (4x the original waypoints) for a smooth visual curve
        num_smooth_points = len(optimized_points) * 4  
        t_raw = np.linspace(0, 1, len(optimized_points))
        t_smooth = np.linspace(0, 1, num_smooth_points)
        
        # Calculate quadratic (k=2) splines for Y and X independently
        spl_y = make_interp_spline(t_raw, optimized_points[:, 0], k=2)(t_smooth)
        spl_x = make_interp_spline(t_raw, optimized_points[:, 1], k=2)(t_smooth)
        
        # Merge, clip to keep the curves safely inside arena boundaries, and return
        smoothed_path = np.column_stack((spl_y, spl_x))
        smoothed_path[:, 0] = np.clip(smoothed_path[:, 0], 2, h_max - 2)
        smoothed_path[:, 1] = np.clip(smoothed_path[:, 1], 2, w_max - 2)
        return smoothed_path
    except Exception as e:
        print(f"Smoothing fallback triggered: {e}")
        return optimized_points
