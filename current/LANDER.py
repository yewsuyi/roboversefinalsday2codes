# 3x swarm drone landing code
import asyncio
import numpy as np
from ScanMap import ScanMapper
from Astar import pathfind
from dola import Dola
from DroneDrivers.HULAXdrone import Drone

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
USE_HULA_MANUAL_MODE = True
USE_UWB_MODE = True
OBS_INFLATION_BUFFER = 10
CASCADE_DELAY = 5.0

#IP:
drone_info = {
    "TODOIP1":{
        "UWB_TAG": 0,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym": -1.38,
        "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm": -2.75,
        "LANDING_Nym": -1.38,
        "LANDING_Exm": -2.75,
        "ORDER":0,
        },

    # "TODOIP2":{
    #     "UWB_TAG": 1,
    #     "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym": -1.38,
    #     "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm": -2.90,
    #     "LANDING_Nym": -1.38,
    #     "LANDING_Exm": -2.90,
    #     "ORDER":1,
    # },

    # "TODOIP3":{
    #     "UWB_TAG": 2,
    #     "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym": -1.38,
    #     "SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm": -3.05,
    #     "LANDING_Nym": -1.38,
    #     "LANDING_Exm": -3.05,
    #     "ORDER":2,
    # }

}
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
ARENA_NORTHLENGTH = 220
ARENA_EASTLENGTH = 110
METRES_PER_SCANMAP_CELL = 0.05
BORDERWALL_THICKNESS = 1

drones = {} #store drone objects here after connecting to them, key by ip address
scanmappers = {}
waypoints_xym = {}
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
                self,
                USE_HULAX_MANUAL_MODE,
                info["UWB_TAG"],
                USE_UWB_MODE,
                ip,
            )
            drones[ip].connect()
            drones[ip].face_camera_down()
            print("drone connected")


            scanmappers[ip] = ScanMapper(
                heightcells_NORTHLENGTH=ARENA_NORTHLENGTH,
                widthcells_EASTLENGTH=ARENA_EASTLENGTH,
                metrespercell=METRES_PER_SCANMAP_CELL,
                ScanmapOriginOffset_Exm=info["SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Exm"],
                ScanmapOriginOffset_Nym=info["SCANMAP_ORIGIN_IN_GLOBAL_COORDS_Nym"],
                OBSTACLEMAP=obstaclemap,
                scanradius=10,
                scanwidth=13,
                borderwallthickness=BORDERWALL_THICKNESS,
            )


            drone_xu, drone_yu = scanmappers[ip].worldNE_to_scanmapXY(0.0, 0.0)
            goal_xu, goal_yu = scanmappers[ip].worldNE_to_scanmapXY(info["LANDING_Nym"], info["LANDING_Exm"])

            path_yxu = pathfind(
                obstaclemap=obstaclemap,
                start_xu=drone_xu,
                start_yu=drone_yu,
                goal_xu=goal_xu,
                goal_yu=goal_yu,
                buffer=OBS_INFLATION_BUFFER,
            ) #list of yxu tuples
            
            path_yxm = [scanmappers[ip].scanmapXY_to_worldNE(x, y) for y, x in path_yxu]
            waypoints_xym[ip] = [(E, N) for N, E in path_yxm]
            #we want XYM coords

            # waypoints = simplifypath(path) #waypoints are a list of tuples(Ny_u, Ex_u)
            # instructions = calc_instructions(waypoints, METRES_PER_CELL) #(direction, Ny_offset, Ex_offset)
            # # if len(instructions)>1:
            # #     if abs(instructions[-1][1]) + abs(instructions[-1][2]) < 0.6:
            # #         #last 2 points is a microadjustment, ignore
            # #         print(f"last instr dist < 0.6, ignoring {instructions[-1][0]}wards: {instructions[-1][1]:.2f}N | {instructions[-1][2]:.2f}")
            # #         instructions.pop()
            # astarinstructions[ip] = instructions

        stagecount = 0
        numdrones = len(drones)
        while stagecount <= numdrones:
            for ip, info in drone_info.items():
                if info["ORDER"] == stagecount:
                    print(f"Stage {stagecount}: Drone {ip} taking off")
                    drones[ip].arm_and_takeoff(False)
                
                elif info["ORDER"] == stagecount - 1:
                    print(f"Stage {stagecount}: Drone {ip} taking off")
                    asyncio.create_task(drones[ip].follow_waypoints_and_land(
                        scanmappers[ip],
                        parser,
                        waypoints_xym[ip],
                        waypoints_xym[ip][-1],
                    ))
                    # await drone.follow_waypoints(
                    #     scanmapper,
                    #     parser,
                    #     path_to_follow_xym, #waypoints_xym,
                    #     state,
                    #     path_to_follow_xym[-1],
                    # )

            stagecount+=1
            await asyncio.sleep(CASCADE_DELAY)
        
        while True: await asyncio.sleep(1.0) #keep thread alive
            

    except Exception as e: raise e
    finally:
        parser.stop()
        parser.join()


if __name__ == "__main__":
    asyncio.run(run())