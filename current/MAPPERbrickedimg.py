# RECON MAPPING CODE
import asyncio
import time
import numpy as np
import math
import traceback

from PointClouder import GlobalMapper, scannow, depthcam_pointcloud_task
from ScanMap import ScanMapper
from Plotdisplay import pointcloud_plotter_task
from UWBaller import UWBParserThread
from Astar import pathfind, simplifypath
from MAPPERPATH import expand_waypoints

# THE BIG 4 MODULAR ONES # +++++++++++++++++++++++
from CameraReceivers.RealsenseCamera import CameraReceiver
from DroneDrivers.MAVdrone import Drone
from PositionGetters.get_position_with_task import SharedState, position_monitor_task
from Imager import imager_task
# THE BIG 4 MODULAR ONES # +++++++++++++++++++++++

# TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE #
SYSTEM_ADDRESS = "serial:///dev/ttyS6:921600" #="udpin://0.0.0.0:14540"
TAG_IG = 0
VALID_ARUCO_IDS = () #TODO TUNE FOR IRL DRONE
# TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE #

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
BRICKEDMODE = True
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym = -1.38
SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm = -2.90
ARENA_NORTHLENGTH = 220
ARENA_EASTLENGTH = 110
METRES_PER_SCANMAP_CELL = 0.05
BORDERWALL_THICKNESS = 1


# TUNABLES # TUNABLES # TUNABLES # TUNABLES # TUNABLES # TUNABLES # TUNABLES #
DRONE_HEIGHT = 2.0 #2.25
ADDITIONAL_HEIGHT = 0.08 #0.20 #overdrive the Pd controller to takeoff faster

OBS_H_MIN = 0.10 # Minimum obstacle height to care about (relative to floor)
OBS_H_MAX = 1.50 # Maximum obstacle height to care about (relative to floor)

SCANS_PER_POINTCLOUDSCAN = 1
POINTCLOUDERDELAY = 1.0
MAPDRAWERDELAY = 3.0
IMAGE_DETECTOR_COOLDOWN = 1.0
# TUNABLES # TUNABLES # TUNABLES # TUNABLES # TUNABLES # TUNABLES # TUNABLES #


# LEGACY - DONT TOUCH
USE_UWB_MODE = False
SCAN_RADIUS_M = 10
SCAN_WIDTH_M = 13
MAP_YAW_ERROR_RAD=0
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

battery_remain = -1
async def battery_task(drone):
    global battery_remain
    async for battery in drone.telemetry.battery():
        battery_remain = battery.remaining_percent

async def run():

    if USE_UWB_MODE:
        parser = UWBParserThread()
        if not parser.serial_port:
            print("No UWB device detected. Exiting.")
            return
        else: parser.start()
    else: parser = None

    receiver = CameraReceiver()
    
    scanmapper = ScanMapper(
        heightcells_NORTHLENGTH=ARENA_NORTHLENGTH,
        widthcells_EASTLENGTH=ARENA_EASTLENGTH,
        metrespercell=METRES_PER_SCANMAP_CELL,
        ScanmapOriginOffset_Exm=SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm,
        ScanmapOriginOffset_Nym=SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym,
        OBSTACLEMAP=None,
        scanradius=SCAN_RADIUS_M,
        scanwidth=SCAN_WIDTH_M,
        borderwallthickness=BORDERWALL_THICKNESS,
    )

    pointclouder = GlobalMapper(
        cam_height=DRONE_HEIGHT, #1.0
        obs_h_min=OBS_H_MIN,
        obs_h_max=OBS_H_MAX,
        depth_min=0.03, #0.3
        depth_max=SCAN_RADIUS_M,
        scans_per_pointcloud=SCANS_PER_POINTCLOUDSCAN,
        yaw_in_degrees=True,
        yaw_smoothing=1.0,
        map_yaw_error_rad=MAP_YAW_ERROR_RAD,
    )

    drone = Drone(
        UWB_TAG=TAG_IG,
        USE_UWB_MODE=False,
        system_address=SYSTEM_ADDRESS,
        takeoff_height=DRONE_HEIGHT,
    )

    #SETUP
    stop_event = asyncio.Event()
    monitor_task = None
    vision_task = None
    pointcloud_visualiser_task = None
    pointcloud_updater_task = None

    try:

        await drone.connect()
        asyncio.create_task(battery_task(drone.drone))

        state = SharedState()
        state.blockpointclouding = True
        monitor_task = asyncio.create_task(position_monitor_task(drone, state, stop_event))
        while not (state.NEDready and state.ATTready and state.ATTVELready):
            print("waiting for telemetry to get online...")
            await asyncio.sleep(0.1)

        if not BRICKEDMODE:
            print(f"STARTING batt:{battery_remain}")
            print("performing light takeoff...")
            await drone.arm_and_takeoff()
            # await drone.prep_offboard(0.0)
            await drone.NEW_prep_offboard()
            print("\n=== TAKING OFF, FRFR ===")

            await drone.wait_for_takeoff_stable(state, ADDITIONAL_HEIGHT)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

        # IMAGE DETECTOR TASK
        vision_task = asyncio.create_task(imager_task(
            receiver, stop_event,
            IMAGE_DETECTOR_COOLDOWN, VALID_ARUCO_IDS
        ))

        # PLOTTER TASK
        # pointcloud_visualiser_task = asyncio.create_task(pointcloud_plotter_task(
        #     pointclouder, state, scanmapper, stop_event, MAPDRAWERDELAY
        # ))

        # DEPTHCAM POINTCLOUDER TASK
        # pointcloud_updater_task = asyncio.create_task(depthcam_pointcloud_task(
        #     drone, receiver, pointclouder, state, scanmapper, stop_event,
        #     POINTCLOUDERDELAY, None, False, 0
        # ))

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #


        if BRICKEDMODE:
            
            print("=======\nbricking drone movement (TEST MODE)...\n=======")
            state.blockpointclouding = False
            while True:
                print(f"N: {state.north} | E: {state.east} | D: {state.down} | Yaw: {state.yaw_deg}")
                await asyncio.sleep(1)

        else:

            print(f"POST-SETUP batt:{battery_remain}")
            print("starting main fly script...\n===========")
            state.blockpointclouding = False
            await megatron(drone, state, receiver, pointclouder, scanmapper, parser, pointcloud_visualiser_task)

    except Exception as e:
        print(f"Main code failed: {e}")
        print("\n--- CRITICAL LOG START ---")
        traceback.print_exc() 
        print("--- CRITICAL LOG END ---\n")
        raise

    finally:

        state.blockpointclouding = True
        await asyncio.sleep(2.0)

        print("\n===============\nSaving maps and shutting down...")

        #SAVE OBSTACLE MAP
        np.save('scanmap.npy', scanmapper.scanmap)
        scanmapper.scanmap[scanmapper.scanmap==0] = 2
        scanmapper.scanmap[scanmapper.scanmap>2] = 1
        scanmapper.scanmap[scanmapper.scanmap==1] = 0
        scanmapper.scanmap[scanmapper.scanmap==2] = 1
        np.save('obstaclemap.npy', scanmapper.scanmap)
        print("Scanmap andObstacle map successfully saved!\n===============")

        if receiver is not None:
            receiver.stop()
        if parser is not None:
            parser.stop()
            parser.join()

        print("\n\n\n============================ MAIN FLY SCRIPT EXITED ============================\n\n\n")
        stop_event.set()

        if monitor_task is not None:
            monitor_task.cancel()
            try: await monitor_task
            except asyncio.CancelledError: print("Position monitor task cancelled.")
        if pointcloud_visualiser_task is not None:
            pointcloud_visualiser_task.cancel()
            try: await pointcloud_visualiser_task
            except asyncio.CancelledError: print("Pcloud visualiser task cancelled.")
        if pointcloud_updater_task is not None:
            pointcloud_updater_task.cancel()
            try: await pointcloud_updater_task
            except asyncio.CancelledError: print("Pcloud updater task cancelled.")
        if vision_task is not None:
            vision_task.cancel()
            try: await vision_task
            except asyncio.CancelledError: print("vision imager task cancelled.")


        await drone.land()
        await asyncio.sleep(30.0) #give time for post mortem









async def megatron(drone, state, receiver, mapper, scanmapper:ScanMapper, parser, pointcloud_visualiser_task): #pilot loop

    wp_xyu = expand_waypoints()

    if pointcloud_visualiser_task is not None:
        xs, ys = zip(*wp_xyu)
        scanmapper.scanmap[list(ys), list(xs)] = 5

    path_to_follow_yxm = [scanmapper.scanmapXY_to_worldNE(X, Y) for X, Y in wp_xyu]
    path_to_follow_xym = [(E, N) for N, E in path_to_follow_yxm]

    print("GO!!!")
    await drone.follow_waypoints(
        scanmapper,
        parser,
        path_to_follow_xym, #waypoints_xym,
        state,
        path_to_follow_xym[-1],
    )

    print(f"HEALTH: {battery_remain}% | h: {-state.down}m")
    print("\n===\nFINISHED NAVIGATION\n===\n")


if __name__ == "__main__":
    asyncio.run(run())
