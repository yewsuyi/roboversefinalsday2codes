import numpy as np

def depth_to_xy_map_downward(
    depth_img,
    height, width, 
    fx, fy, cx, cy, 
    depth_scale=1, 
    dronecam_height=1.0, 
    obs_h_min=0.05,        # Minimum obstacle height to care about (relative to floor)
    obs_h_max=1.5,         # Maximum obstacle height to care about (relative to floor)
    MIN_DEPTH=0.1,         # Keeps sensor clipping noise clean
    MAX_DEPTH=5.0,         # Max downrange ceiling threshold
):
    """
    Pinhole projection mapper tailored specifically for a DOWNWARD-facing depth camera.
    Returns body-frame X (Right) and Z (Forward) point cloud arrays.
    """
    MIN_DEPTH=0.1 #FORCE VALUE
    MAX_DEPTH=5.0 #FORCE VALUE

    if (not height) or (not width): 
        height, width = depth_img.shape

    u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
    u_coords = u_coords.astype(np.float32)
    v_coords = v_coords.astype(np.float32)

    depth_m = depth_img.astype(np.float32) * depth_scale

    # 1. Filter out raw out-of-bounds camera signal dropouts
    valid_sensor_range = (depth_m > MIN_DEPTH) & (depth_m < MAX_DEPTH)

    # 2. Pinhole Deprojection into Drone Body Space
    # Camera Z is down. Therefore, depth_m IS the distance below the camera.
    y_body = depth_m  

    # Camera X is right, matching X_body
    x_body = (u_coords - cx) * y_body / fx
    
    # Camera Y is backward, so Z_body (Forward) is inverted Camera Y
    z_body = -(v_coords - cy) * y_body / fy

    # 3. Height Band Filter (Translate to absolute distance from the floor)
    # y_body is the distance DOWN from the drone camera.
    # Therefore: Object Height Above Floor = dronecam_height - y_body
    object_height_above_floor = dronecam_height - y_body

    valid_height_band = (
        (object_height_above_floor >= obs_h_min) & 
        (object_height_above_floor <= obs_h_max)
    )

    combined_valid_mask = valid_sensor_range & valid_height_band

    # Extract flat coordinate arrays of valid obstacle hits
    X = x_body[combined_valid_mask]
    Z = z_body[combined_valid_mask]

    return X, Z


def depth_pixel_to_xz_downward(
    depth_img,
    u, #column - x
    v, #row - y
    fx, fy, cx, cy,
    depth_scale,
):
    try:
        z_cam = float(depth_img[v, u]) * depth_scale

        if z_cam <0.0: return None, None

        y_body = z_cam
        x_body = (u - cx) * y_body / fx
        z_body = -(v - cy) * y_body / fy

        return x_body, z_body

    except Exception as e: return None, None









def depth_pixel_to_xz(
    depth_img,
    u, #column - x
    v, #row - y
    fx, fy, cx, cy,
    depth_scale,
    use_pitchdown,
    pitchdown,
):
    try:
        z_cam = float(depth_img[v, u]) * depth_scale

        x_cam = (u - cx) * z_cam / fx

        if not use_pitchdown: return x, z
        # --- #
        y_cam = (v - cy) * z_cam / fy

        pitch_rad = np.deg2rad(pitchdown)
        c = np.cos(pitch_rad)
        s = np.sin(pitch_rad)

        x_body = x_cam
        # y_body = y_cam * c + z_cam * s
        z_body = z_cam * c - y_cam * s

        return x_body, z_body

    except Exception as e: return None, None





def depth_to_xy_map_with_pitchdown(
    depth_img,

    height, width, #camera intrinsics, take from camera receiver
    fx, fy, cx, cy, #camera intrinsics, take from camera receiver
    depth_scale=1, #camera intrinsics, take from camera receiver

    dronecam_height=1.0, #current drone height
    obs_h_min=0.05, #INCREASE TO NOT SCAN THE FLOOR
    obs_h_max=1.5, #DECREASE TO NOT SCAN THE CEILING
    MIN_DEPTH=0.2,
    MAX_DEPTH=15.0, #set to scanradius
    pitchdown=0.0, #camera downward tilt in degrees

):
    """
    Returns coordinate np arrays X and Z which correspond to the pointcloud.
    This version corrects for a camera that is pitched downward.

    Coordinate convention:
    - raw camera X points right
    - raw camera Y points down
    - raw camera Z points forward
    - pitchdown is positive when the camera is tilted downward
    """

# COMPUTE MESH # COMPUTE MESH # COMPUTE MESH #
    if (not height) or (not width): height, width = depth_img.shape

    u_coords, v_coords = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )
    u_coords = u_coords.astype(np.float32)
    v_coords = v_coords.astype(np.float32)
# COMPUTE MESH # COMPUTE MESH # COMPUTE MESH #

    depth_m = depth_img.astype(np.float32) * depth_scale

    valid = (
        (depth_m > MIN_DEPTH) &
        (depth_m < MAX_DEPTH)
    )

    # ----------------------------------
    # Deproject pixels -> raw camera 3D
    # ----------------------------------

    z_cam = depth_m
    x_cam = (u_coords - cx) * z_cam / fx
    y_cam = (v_coords - cy) * z_cam / fy

    # ----------------------------------
    # Correct for camera pitch
    # Positive pitchdown means the camera is tilted downward.
    # This rotates camera-space points back into drone/body local space.
    # ----------------------------------

    pitch_rad = np.deg2rad(pitchdown)
    c, s = np.cos(pitch_rad), np.sin(pitch_rad)

    x_body = x_cam
    y_body = y_cam * c + z_cam * s
    z_body = z_cam * c - y_cam * s

    # ----------------------------------
    # Height band filter (remove ground & ceiling)
    # In corrected body frame: Y points DOWN.
    # Ground is at y = dronecam_height.
    # ----------------------------------

    y_min_body = dronecam_height - obs_h_max
    y_max_body = dronecam_height - obs_h_min

    valid_h = (
        (y_body >= y_min_body) &
        (y_body <= y_max_body)
    )

    valid_forward = z_body > 0

    combined_valid = valid & valid_h & valid_forward

    # Flatten valid points
    X = x_body[combined_valid]
    Z = z_body[combined_valid]

    return X, Z














def depth_to_xy_map(
    depth_img,

    height, width, #camera intrinsics, take from camera receiver
    fx, fy, cx, cy, #camera intrinsics, take from camera receiver
    depth_scale=1, #camera intrinsics, take from camera receiver

    dronecam_height=1.0, #current drone height
    obs_h_min=0.05, #INCREASE TO NOT SCAN THE FLOOR
    obs_h_max=1.5, #DECREASE TO NOT SCAN THE CEILING
    MIN_DEPTH=0.2,
    MAX_DEPTH=15.0, #set to scanradius

):
    """
    Returns coordinate np arrays X and Z which correspond to the pointcloud
    - each point's x,z coord is X[i],Z[i]
    """

# COMPUTE MESH # COMPUTE MESH # COMPUTE MESH #
    # reference
    # h, w = depth_img.shape
    # u, v = np.meshgrid(np.arange(w), np.arange(h), indexing='xy') #indexing=xy should be default behaviour

    if (not height) or (not width): height, width = depth_img.shape

    u_coords, v_coords = np.meshgrid(
        np.arange(width),
        np.arange(height)
    )
    u_coords = u_coords.astype(np.float32)
    v_coords = v_coords.astype(np.float32)
# COMPUTE MESH # COMPUTE MESH # COMPUTE MESH #

    depth_m = depth_img.astype(np.float32) * depth_scale

    valid = (
        (depth_m > MIN_DEPTH) &
        (depth_m < MAX_DEPTH)
    )

    # ----------------------------------
    # Deproject pixels -> 3D
    # ----------------------------------

    z = depth_m
    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy

    # ----------------------------------
    # Height band filter (remove ground & ceiling)
    # In optical frame: Y points DOWN. Ground is at y = dronecam_height.
    # ----------------------------------

    y_min_cam = dronecam_height - obs_h_max
    y_max_cam = dronecam_height - obs_h_min

    valid_h = (
        (y >= y_min_cam) &
        (y <= y_max_cam)
    )

    combined_valid = valid & valid_h

    # Flatten valid points
    X = x[combined_valid]
    Z = z[combined_valid]

    return X, Z

def local_to_ned_global(X, Z, north, east, yaw_rad):
    """Returns NE global (north, east) NP ARRAYS from local (X_cam=right, Z_cam=forward) np arrays"""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    X_new = Z * c - X * s
    Y_new = Z * s + X * c
    # return np.column_stack([north + X_new, east + Y_new])
    return north + X_new, east + Y_new

# #from GlobalMapperV2
# def _local_to_ned_global(local_xy, north, east, yaw_rad):
#     # print(f"post-depth-img coords: {north} | {east} | {yaw_rad}")

#     """Transform local (X_cam=right, Z_cam=forward) to NED global (north, east)"""
#     X_cam = local_xy[:, 0]
#     Z_cam = local_xy[:, 1]
#     X = Z_cam
#     Y = X_cam

#     c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    
#     X_new = X * c - Y * s
#     Y_new = X * s + Y * c
#     return np.column_stack([north+X_new, east+Y_new])