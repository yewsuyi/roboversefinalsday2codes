# position async reporter task but for pyhulax
from pyhulax import DroneAPI








# BIG ALERT! THIS FILE IS OBSOLETE
# AS PYHULAX DRONES DO NOT NEED THE SHAREDSTATE VARIABLE AT ALL
# TO GET THE DRONE'S POSITION NED, CALL (await) HULAXdrone.py's await Drone.get_position_ned()




























from enum import IntEnum

class Direction(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3

class SharedState:
    """Thread-safe(ish) container for inter-task data in a single event loop."""
    def __init__(self):
        self.latest_position = None  # NED position from SharedState
        self.roll = None
        self.pitch = None
        self.is_armed = False
        self.control_active = False

        # self.NEDready = False
        self.north = None
        self.east = None
        self.down = None
        # self.north_velo = None
        # self.east_velo = None
        # self.down_velo = None

        # #for compatibility with 
        # state.ATTready = True
        # state.ATTVELready = True


async def position_monitor_task(drone: Drone, state: SharedState, stop_event: asyncio.Event):
    """
    Background task streaming NED position and Yaw updates concurrently.
    """
    print("Position monitor task started...")

    try:
        async for pos_vel in drone.drone.telemetry.position_velocity_ned():
            if stop_event.is_set(): break
            state.north = pos_vel.position.north_m
            state.east = pos_vel.position.east_m
            state.down = pos_vel.position.down_m
            # state.north_velo = pos_vel.velocity.north_m_s
            # state.east_velo = pos_vel.velocity.east_m_s
            # state.down_velo = pos_vel.velocity.down_m_s
            # state.NEDready = True

    except asyncio.CancelledError: print("📡 Position monitor task cancelled.")
    except Exception as e: print(f"Monitor error: {type(e).__name__}: {e}")
