# 3x swarm drone landing code
import asyncio
import numpy as np
from Astar import pathfind

safe_entrypoints_XYU = [] # one tuple coord for each drone
landing_points_XYU = [] # one tuple coord for each drone
astarinstructions = []



NUM_DRONES = 3

#CHECK
METRES_PER_CELL = 0.1 #MUST AGREE ACROSS MAPPER/LANDER/SHOCKWAVE.py
OBS_INFLATION_BUFFER = 10

async def run():

    # load obstaclemap, a boolean array
    obstaclemap = np.load("obstaclemap.npy")

    # generate astar path instructions
    for i in range(NUM_DRONES):

        path = pathfind(
            obstaclemap=obstaclemap,
            start_xu=safe_entrypoints_XYU[i][0], #or the drone's position's XU (convert)
            start_yu=safe_entrypoints_XYU[i][1], #or the drone's position's YU (convert)
            goal_xu=landing_points_XYU[i][0],
            goal_yu=landing_points_XYU[i][1],
            buffer=OBS_INFLATION_BUFFER,
        )

        waypoints = simplifypath(path) #waypoints are a list of tuples(Ny_u, Ex_u)
        instructions = calc_instructions(waypoints, METRES_PER_CELL) #(direction, Ny_offset, Ex_offset)
        # if len(instructions)>1:
        #     if abs(instructions[-1][1]) + abs(instructions[-1][2]) < 0.6:
        #         #last 2 points is a microadjustment, ignore
        #         print(f"last instr dist < 0.6, ignoring {instructions[-1][0]}wards: {instructions[-1][1]:.2f}N | {instructions[-1][2]:.2f}")
        #         instructions.pop()

        astarinstructions.append(instructions)


    # TODO DRONE SWARM SETUP - refer to huladola.py
    # MAKE ALL 3 DRONES FLY UP TOGETHER, drone.arm_and_takeoff(blocking=False)
    # OR drones fly up one by one, drone.arm_and_takeoff(blocking=True)





    # for each drone, go to safe entrypoint, then follow boxy astar path to the landing zone, then land
    await goto_entrypoint(drones, 0)

    while True: asyncio.sleep(1) #keep thread awake?

async def goto_entrypoint(drones, droneno):

    if droneno >= NUM_DRONES: return


    #TODO GOTO ENTRYPOINT



    follow_path_found(drones, droneno)
    # asyncio.sleep(1.0)
    goto_entrypoint(drones, droneno+1)

async def follow_path_found(drones, droneno):

    # path is astarinstructions[droneno]

    print(f"MOVING DRONE [{droneno}] NOW...")
    leninst = len(instructions)
    for i in range(leninst):

        print(f"move {i+1}/{leninst} - {instructions[i][0]}wards: {instructions[i][1]:.2f}N | {instructions[i][2]:.2f}E")
        #TODO MOVE (OFFSET POSITION) BY await drone.goto_NEpos(state, instructions[i][1], instructions[i][2])

    #TODO LAND DRONE




def calc_instructions(path, resolution=0.1): #path is y,x

    instructions = []
    if len(path)<2: return path

    for i in range(len(path) - 1):

        dy = path[i+1][0] - path[i][0]
        dx = path[i+1][1] - path[i][1]
        dir = ""

        if dy == 0: #horizontal movement (x or E-W)
            if dx > 0: dir = "east"
            else: dir = "west"
        else: #vertical movement (y or N-S)
            if dy > 0: dir = "north" 
            else: dir = "south"

        instructions.append((dir, dy*resolution, dx*resolution)) 

    #(direction to turn to, Ny_offset, Ex_offset)
    #read direction to know how to turn, feed Ny_offset, Ex_offset into drone.goto_NEpos(Ny, Ex) directly
    return instructions

if __name__ == "__main__":
    asyncio.run(run())
