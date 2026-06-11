# DRONE SWARM AMBUSH CODE
import numpy as np
from PatrolPathGen.main import init, waypoints

async def run():

    obstaclemap = np.load("obstaclemap.npy")
    obstaclemap[obstaclemap==0] = 2
    obstaclemap[obstaclemap==1] = 0
    obstaclemap[obstaclemap==2] = 1

    #swap the 2 bc yw's code needs it swapped
    init(obstaclemap)

    while True:




if __name__ == "__main__":
    asyncio.run(run())