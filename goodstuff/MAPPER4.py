# RECON MAPPING CODE
import asyncio
import time
import numpy as np
import math
import traceback
# from plotarray import plot_array

from PointClouder import GlobalMapper, scannow, depthcam_pointcloud_task
from ScanMap import ScanMapper
from Plotdisplay import pointcloud_plotter_task
from UWBaller import UWBParserThread
from Astar import pathfind, simplifypath
# from pathextractor import extract_path
from MAPPERPATH import expand_waypoints

'''
commander set_ekf_origin 47.397742 8.545594 488.0
'''

# THE BIG 3 MODULAR ONES # TODO
from CameraReceivers.RealsenseCamera import CameraReceiver
from DroneDrivers.MAVdrone import Drone
from PositionGetters.get_position_with_task import SharedState, position_monitor_task
# from Imager import imager_task
# THE BIG 3 MODULAR ONES # TODO

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

BRICKEDMODE = False # NOTE TEST MODEs

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #

# TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE #
SYSTEM_ADDRESS = "serial:///dev/ttyS6:921600"
TAG_IG = 0
USE_UWB_MODE = False
USE_INFRARED = False
VALID_ARUCO_IDS = () #TODO TUNE FOR IRL DRONE
# SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"

SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym = -1.38
SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm = -2.90
ARENA_NORTHLENGTH = 220 #TODO COMP SETTING
ARENA_EASTLENGTH = 110 #TODO COMP SETTING
METRES_PER_SCANMAP_CELL = 0.05

FRONTIER_STRATEGY = 0
#0: furthest from origin
#1: furthest from dronepos (this is prob bad dont use)
#2: closest to dronepos
INITIAL_SAFEPRINT_RADIUS_M = 0.8 #must be bigger than inflation
# TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE # TODO CHANGE #
BORDERWALL_THICKNESS = 1


#ORIGINAL
# DRONE_HEIGHT = 3.25

# OBS_H_MIN = 2.80 ####2.75 #2.95 ##3.0 # increasing this makes the floor go away #TODO TUNE
# OBS_H_MAX = 2.85 ####2.85 #3.05 ##3.10 #3.25 #lowering this makes the propellers go away #TODO REVERT TO 3.25 FOR IRL DRONE

# SCAN_RADIUS_M = 10
# SCAN_WIDTH_M = 13
#ORIGINAL

#NEW
DRONE_HEIGHT = 2.0
ADDITIONAL_HEIGHT = 0.05 #overdrive the Pd controller to takeoff faster

OBS_H_MIN = 0.5 # #1.0 #1.80# increasing this makes the floor go away #TODO TUNE FOR IRL DRONE
OBS_H_MAX = DRONE_HEIGHT # #1.85 #lowering this makes the propellers go away #TODO REVERT TO 2.25 FOR IRL DRONE

SCAN_RADIUS_M = 10
SCAN_WIDTH_M = 13 

USE_PITCHDOWN = False
PITCHDOWN = 20.0
#NEW

SCANS_PER_POINTCLOUDSCAN = 1
POINTCLOUDERDELAY = 1.0
MAPDRAWERDELAY = 3.0
IMAGE_DETECTOR_COOLDOWN = 1.0

SAFETY_PAUSE = 0.8
BLOCKYPATHMODE = True
OBS_INFLATION_BUFFER=20
DRONE_DOMAIN_BUFFER=15

MAP_YAW_ERROR_RAD=0


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

    #1.0 and 1.5 still detect the barrels but is more clean
    pointclouder = GlobalMapper(
        cam_height=DRONE_HEIGHT, #1.0
        obs_h_min=OBS_H_MIN,
        obs_h_max=OBS_H_MAX,
        depth_min=0.03, #0.3
        depth_max=SCAN_RADIUS_M,
        use_pitchdown=USE_PITCHDOWN,
        pitchdown=PITCHDOWN,

        scans_per_pointcloud=SCANS_PER_POINTCLOUDSCAN,

        yaw_in_degrees=True,
        yaw_smoothing=1.0,
        map_yaw_error_rad=MAP_YAW_ERROR_RAD,
    )

    drone = Drone(
        UWB_TAG=TAG_IG,
        USE_UWB_MODE=False, #NOTE NOTE NOTE
        system_address=SYSTEM_ADDRESS,
        takeoff_height=DRONE_HEIGHT,
    )

    #SETUP
    stop_event = asyncio.Event()
    monitor_task = None
    pointcloud_visualiser_task = None
    pointcloud_updater_task = None
    vision_task = None


    #SETUP
    try:

        # connect and wait for pre-arm readiness inside Drone wrapper
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


    # # IMAGE DETECTOR TASK - IMAGE DETECTOR TASK - IMAGE DETECTOR TASK - IMAGE DETECTOR TASK #TODO
    #     vision = VisionApp(IMAGE_DETECTOR_COOLDOWN)
        #vision_task = asyncio.create_task(vision.run())
        # vision_task = asyncio.create_task(imager_task(
        #     receiver, mapper, state, parser, scanmapper,
        #     USE_UWB_MODE, USE_PITCHDOWN, PITCHDOWN,
        #     stop_event, IMAGE_DETECTOR_COOLDOWN,
        #     VALID_ARUCO_IDS, USE_INFRARED
        # ))
    # # IMAGE DETECTOR TASK - IMAGE DETECTOR TASK - IMAGE DETECTOR TASK - IMAGE DETECTOR TASK #TODO


    # POINTCLOUD & SCANMAP PLOTTER TASK - POINTCLOUD & SCANMAP PLOTTER TASK - POINTCLOUD & SCANMAP PLOTTER TASK
        pointcloud_visualiser_task = asyncio.create_task(pointcloud_plotter_task(
            pointclouder, state, scanmapper, stop_event,
            drawloopdelay=MAPDRAWERDELAY
        ))
    # POINTCLOUD & SCANMAP PLOTTER TASK - POINTCLOUD & SCANMAP PLOTTER TASK - POINTCLOUD & SCANMAP PLOTTER TASK


    # DEPTHCAM POINTCLOUDER TASK - DEPTHCAM POINTCLOUDER TASK - DEPTHCAM POINTCLOUDER TASK - DEPTHCAM POINTCLOUDER TASK
        pointcloud_updater_task = asyncio.create_task(depthcam_pointcloud_task(
            drone, receiver, pointclouder, state, scanmapper, stop_event,
            loopdelay=POINTCLOUDERDELAY, #0.5 #1.0 #TODO TUNE
            parser=None, USE_UWB_MODE=False,
            )
        )
    # DEPTHCAM POINTCLOUDER TASK - DEPTHCAM POINTCLOUDER TASK - DEPTHCAM POINTCLOUDER TASK - DEPTHCAM POINTCLOUDER TASK


        if BRICKEDMODE:
            print("=======\nbricking drone movement (TEST MODE)...\n=======")
            state.blockpointclouding = False
            while True:
                print(f"N: {state.north} | E: {state.east}")
                await asyncio.sleep(1)


        print(f"POST-SETUP batt:{battery_remain}")
        print("starting main fly script...\n===========")
        state.blockpointclouding = False
        if not BRICKEDMODE: await megatron(drone, state, receiver, pointclouder, scanmapper, parser)

    except Exception as e:
        print(f"Main code failed: {e}")
        print("\n--- CRITICAL LOG START ---")
        traceback.print_exc() 
        print("--- CRITICAL LOG END ---\n")
        raise

    finally:

        state.blockpointclouding = True

        await asyncio.sleep(10.0) #TODO REMOVE

        #SAVE OBSTACLE MAP
        np.save('scanmap.npy', scanmapper.scanmap)
        scanmapper.scanmap[scanmapper.scanmap==0] = 2
        scanmapper.scanmap[scanmapper.scanmap>2] = 1
        scanmapper.scanmap[scanmapper.scanmap==1] = 0
        scanmapper.scanmap[scanmapper.scanmap==2] = 1
        np.save('obstaclemap.npy', scanmapper.scanmap)
        print("Scanmap andObstacle map successfully saved!")

        receiver.stop()
        if parser is not None:
            parser.stop()
            parser.join()

        print("\n\n\n MAIN FLY SCRIPT EXITED ========================================================= \n\n\n")
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

        # if pilot_task is not None:
        #     pilot_task.cancel()
        #     try: await pilot_task
        #     except asyncio.CancelledError: print("Pilot task cancelled.")



        await asyncio.sleep(30.0) #give time for post mortem
        try:
            # await drone.goto_position(47.397742, 8.545594, 488.0) # home the drone
            await drone.land()
        except Exception as e: print(f"CATASTROPHIC FAILURE: landing skipped or failed: {e}")




async def megatron(drone, state, receiver, mapper, scanmapper:ScanMapper, parser): #pilot loop


    # print("moving drone north...")
    # # await drone.fly_to_position(parser, 5.0, -0.5, state, 0.0)

    # for _ in range(60):
    #     await drone.flystraight(state, 0)
    #     await asyncio.sleep(0.05)

    # print("\nMEGATR0N!!! TEST CONCLUDED\n")
    # await asyncio.sleep(20.0)
    # return

    # await drone.turn_to_yaw_deg(state, 0.0)
    # await drone.turn_to_yaw_deg(state, 90.0)    
    
        # if drone.uwb_mode:

        #     current_n, current_e, valid = drone.get_uwb_position_NE(parser)

        #     if valid is None:
        #         print("UWB data MISSING, cannot navigate.")
        #         await drone.send_velocity(0.0, 0.0, 0.0, state.yaw_deg)  # Stop movement if UWB data is not ready
        #         UWBskipped = True
        #         await asyncio.sleep(0.05)
        #         continue

        #     elif not valid:
        #         print("UWB data OUTDATED, cannot navigate.")
        #         await drone.send_velocity(0.0, 0.0, 0.0, state.yaw_deg)  # Stop movement if UWB data is not ready
        #         UWBskipped = True
        #         await asyncio.sleep(0.05)
        #         continue

        # else:
        #     current_n = state.north
        #     current_e = state.east

        #waypoints = [scanmapper.scanmapXY_to_worldNE(wyEx_u, wyNy_u) for wyNy_u, wyEx_u in path]
        #waypoints_xym = [(E, N) for N, E in waypoints]

        #HAND OVER CONTROL TO drone.follow_waypoints()

    wp_xyu = expand_waypoints()
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
    print("\n===\nFINISHED\n===\n")
    #await asyncio.sleep(SAFETY_PAUSE)


if __name__ == "__main__":
    asyncio.run(run())