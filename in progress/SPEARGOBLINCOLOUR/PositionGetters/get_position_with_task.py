import asyncio
# from Ywidrone import Drone  # Your provided class
import numpy as np
import math
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

        self.NEDready = False
        self.north = None
        self.east = None
        self.down = None
        # self.north_velo = None
        # self.east_velo = None
        # self.down_velo = None

        self.ATTready = False
        self.yaw_deg = None
        self.yaw_rad = None #depends on yaw_deg
        
        self.ATTVELready = False
        self.yaw_deg_angvelo = None

        self.blockpointclouding = False
        self.stuckcount = 0
        self.strictmode = True

        #for pilot
        self.explorespeed = 0.8
        self.exploredirection = None #DIRECTION ENUM
        self.leftspace = 40.0
        self.midspace = 40.0
        self.rightspace = 40.0

        #for dirprinter
        self.check_for_double_scanned = False
        self.curr_CFDS_task = None

async def position_monitor_task(drone, state: SharedState, stop_event: asyncio.Event):
    """
    Background task streaming NED position and Yaw updates concurrently.
    """
    print("Position monitor task started...")

    async def stream_position():
        async for pos_vel in drone.drone.telemetry.position_velocity_ned():
            if stop_event.is_set(): break
            state.north = pos_vel.position.north_m
            state.east = pos_vel.position.east_m
            state.down = pos_vel.position.down_m
            state.north_velo = pos_vel.velocity.north_m_s
            state.east_velo = pos_vel.velocity.east_m_s
            state.down_velo = pos_vel.velocity.down_m_s
            state.NEDready = True

    async def stream_orientation():
        async for att in drone.drone.telemetry.attitude_euler():
            if stop_event.is_set(): break
            state.yaw_deg = att.yaw_deg
            # print(f"degyaw: {state.yaw_deg}")
            # state.yaw_deg = att.yaw_deg + state.true_yaw_deg_offset #NEW
            state.roll = att.roll_deg
            state.pitch = att.pitch_deg
            # state.w, state.x, state.y, state.z = euler_to_quaternion(state.roll, state.pitch, state.yaw_deg) #TEST COMMENTING THIS OUT
            state.yaw_rad = np.deg2rad(state.yaw_deg)
            state.ATTready = True

    async def stream_orientation_OMEGA():
        async for att_angularvelo in drone.drone.telemetry.attitude_angular_velocity_body():
            if stop_event.is_set(): break
            state.yaw_deg_angvelo = math.degrees(att_angularvelo.yaw_rad_s)
            state.ATTVELready = True

    try:
        # Run both streams concurrently
        await asyncio.gather(stream_position(), stream_orientation(), stream_orientation_OMEGA())

    except asyncio.CancelledError: print("📡 Position monitor task cancelled.")
    except Exception as e: print(f"Monitor error: {type(e).__name__}: {e}")