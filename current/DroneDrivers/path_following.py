import math

def global_ned_to_body_velocity(target_north_vel, target_east_vel, current_drone_yaw_deg):
    """
    Transforms global NED velocity commands into drone Body-centric velocities
    (Forward/Right) when the drone's yaw cannot be rotated.
    
    yaw: 0 = north, 90 = east
    """
    yaw_rad = math.radians(current_drone_yaw_deg)
    
    # Rotation matrix project mapping
    forward_vel = (target_north_vel * math.cos(yaw_rad)) + (target_east_vel * math.sin(yaw_rad))
    right_vel = (-target_north_vel * math.sin(yaw_rad)) + (target_east_vel * math.cos(yaw_rad))
    
    return forward_vel, right_vel

def forward_speed_to_ned_velocity(forward_speed, yaw_deg):
    """
    Convert forward speed + yaw to NED velocity.

    yaw:
        0 = north
        90 = east

    Returns:
        north_velocity, east_velocity
    """
    yaw_rad = math.radians(yaw_deg)

    north_velocity = forward_speed * math.cos(yaw_rad)
    east_velocity = forward_speed * math.sin(yaw_rad)

    return north_velocity, east_velocity

def normalize_angle_deg(angle):
    """Keep angle in [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def distance_2d(a, b):
    """Distance between two (x, y) points."""
    return math.hypot(b[0] - a[0], b[1] - a[1])


def bearing_to_target_deg(current_pos, target_pos):
    """
    Get yaw needed to face target.
    x = east, y = north.
    yaw: 0 = north, 90 = east.
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    return math.degrees(math.atan2(dx, dy))


def compute_path_follow_command(
    waypoints,
    drone_position,
    drone_yaw_deg,
    goal_position,
    waypoint_tolerance=0.25,
    goal_tolerance=0.30,
    slowdown_distance=1.0,
    use_yaw_slowdown=True,
    slow_yaw_error_deg=45.0,
    stop_yaw_error_deg=90.0,
    lookahead=1.0,
):
    """
    Follow a waypoint path.

    waypoints:
        Mutable waypoint list. Use mypath = waypoints.copy() before calling
        this in a loop if you need to preserve the original path.

    drone_position:
        Current drone position as (x, y), where x=east, y=north.

    Returns:
        speed_multiplier: 0.0 to 1.0
        target_yaw_deg: yaw to face
        distance_to_goal: metres from final goal
    """
    distance_to_goal = distance_2d(drone_position, goal_position)

    if distance_to_goal <= goal_tolerance:
        return 0.0, drone_yaw_deg, distance_to_goal, goal_position

    if waypoints is None:
        return 0.0, drone_yaw_deg, distance_to_goal, goal_position

    while waypoints:
        if distance_2d(drone_position, waypoints[0]) <= waypoint_tolerance:
            waypoints.pop(0)
        else:
            break

    # target_position = waypoints[0] if waypoints else goal_position
    target_position = get_lookahead_target(waypoints, drone_position, goal_position, lookahead=lookahead)
    target_yaw_deg = bearing_to_target_deg(drone_position, target_position)

    yaw_error = normalize_angle_deg(target_yaw_deg - drone_yaw_deg)
    distance_to_target = distance_2d(drone_position, target_position)

    # if slowdown_distance <= 0:
    #     speed_multiplier = 1.0
    # else:
    #     speed_multiplier = min(1.0, distance_to_target / slowdown_distance)

    speed_multiplier = 1.0
    if use_yaw_slowdown:
        abs_yaw_error = abs(yaw_error)

        if abs_yaw_error >= stop_yaw_error_deg:
            speed_multiplier = 0.0
            # print("stopping, turn too wide")
        elif abs_yaw_error >= slow_yaw_error_deg:
            speed_multiplier = 1.0 - abs_yaw_error/stop_yaw_error_deg
            # print("slowing down, turn is wide")

    return speed_multiplier, target_yaw_deg, distance_to_goal, target_position


def slew_yaw(current_yaw_deg, target_yaw_deg, max_rate_deg_per_tick):
    """
    Step current_yaw toward target_yaw by at most max_rate_deg_per_tick.
    """
    error = normalize_angle_deg(target_yaw_deg - current_yaw_deg)
    step  = max(-max_rate_deg_per_tick, min(max_rate_deg_per_tick, error))
    return normalize_angle_deg(current_yaw_deg + step)

def get_lookahead_target(waypoints, drone_position, goal_position, lookahead=1.0):
    """
    Walk along the waypoint path and return the point that is
    lookahead metres ahead of the drone's current position along the path.
    """
    if not waypoints:
        return goal_position

    # Build the path segments starting from drone position
    path = [drone_position] + waypoints

    accumulated = 0.0
    for i in range(len(path) - 1):
        seg_start = path[i]
        seg_end   = path[i + 1]
        seg_len   = distance_2d(seg_start, seg_end)

        if accumulated + seg_len >= lookahead:
            # Lookahead point is on this segment
            remaining = lookahead - accumulated
            t = remaining / seg_len
            lx = seg_start[0] + t * (seg_end[0] - seg_start[0])
            ly = seg_start[1] + t * (seg_end[1] - seg_start[1])
            return (lx, ly)

        accumulated += seg_len

    # Lookahead extends past all waypoints — target the goal
    return goal_position