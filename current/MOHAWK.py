# 3x swarm drone landing code
import asyncio
import numpy as np
from ScanMap import ScanMapper
from Astar import pathfind
from dola import Dola
from DroneDrivers.HULAXdrone import Drone
from pyhulax.core import Direction, TelemetryUnavailable,CameraPitchMode,VelocityLevel
from UWBaller import UWBParserThread
from hulaImager import detect_aruco_markers
import random

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
USE_HULAX_MANUAL_MODE = True
USE_UWB_MODE = False
OBS_INFLATION_BUFFER = 10
CASCADE_DELAY = 5.0

#TODO TODO
TODOIP1 = ""
TODOIP2 = ""
TODOIP3 = ""

SOIGC_Nym1 = -1.38
SOIGC_Exm1 = -2.75
SOIGC_Nym2 = -1.38
SOIGC_Exm2 = -2.90
SOIGC_Nym3 = -1.38
SOIGC_Exm3 = -3.05


drone_info = {
    TODOIP1:{
        "UWB_TAG": 0,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym": SOIGC_Nym1,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm": SOIGC_Exm1,
        "LANDING_Nym": 4.4-SOIGC_Nym1,
        "LANDING_Exm": 1.35-SOIGC_Exm1,
        "ORDER":0,
        },

    TODOIP2:{
        "UWB_TAG": 1,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym": SOIGC_Nym2,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm": SOIGC_Exm2,
        "LANDING_Nym": 7.85-SOIGC_Nym2,
        "LANDING_Exm": 1.3-SOIGC_Exm2,
        "ORDER":1,
    },

    TODOIP3:{
        "UWB_TAG": 2,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym": SOIGC_Nym3,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm": SOIGC_Exm3,
        "LANDING_Nym": 4.4-SOIGC_Nym3,
        "LANDING_Exm": 4.4-SOIGC_Exm3,
        "ORDER":2,
    }

}
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
ARENA_NORTHLENGTH = 220
ARENA_EASTLENGTH = 110
METRES_PER_SCANMAP_CELL = 0.05
BORDERWALL_THICKNESS = 1

IMAGERCOOLDOWN = 0.25

drones = {} #store drone objects here after connecting to them, key by ip address
scanmappers = {}
waypoints_xym = {}
asyncscanners = {}
# astarinstructions = {}

async def run():

    try: 

        if USE_UWB_MODE:
            parser = UWBParserThread()
            if not parser.serial_port:
                print("No UWB device detected. Exiting.")
                return
            else: parser.start()
        else: parser = None

        # load obstaclemap, a boolean array
        obstaclemap = np.load("obstaclemap.npy")

        # generate astar path instructions
        for ip, info in drone_info.items():


            drones[ip] = Drone(
                USE_HULAX_MANUAL_MODE,
                info["UWB_TAG"],
                USE_UWB_MODE,
                ip,
            )
            drones[ip].connect()
            drones[ip].face_camera_down()
            print("drone connected")

            asyncscanners[ip] = asyncio.create_task(detect_aruco_markers(
                drone_info[ip]["ORDER"],
                drones[ip],
                IMAGERCOOLDOWN,
                ))

        drones[TODOIP1].arm_and_takeoff(True)
        drones[TODOIP2].arm_and_takeoff(True)
        drones[TODOIP3].arm_and_takeoff(True)

        while True:

            for ip, info in drone_info.items():
                randy = random.randint(-50, 850)
                randx = random.randint(-100, 100)
                # drones.drone.fly_to(randx, randy, 100)
                drones[ip].drone.move_to(x=randx, y=randy, z=100, speed=VelocityLevel.SLOW)



            await asyncio.sleep(5.0)

    except Exception as e: raise e
    finally:
        # for ip, info in drone_info.items():
        #     drones[ip].land(False)
        if parser is not None:
            parser.stop()
            parser.join()
            