import asyncio
from dola import Dola

async def run():

    dola = Dola()
    dola.start()
    try:
        print("Searching for all drones")
        d = dola.get_all_ips(listen_seconds=5)
    finally: dola.stop()

    drones = {} # store all drones object for control

    for plane_id, ip in d.items():
        print(f"Plane {plane_id}: {ip}")
        drones[str(ip)] = DroneAPI()

        drones[str(ip)].connect(ip) # connect to ip address to gain control of drone
        drones[str(ip)].face_camera_down()
        # drones[str(ip)].BLINK/LIGHTUP #YUJUN TODO
        print(f"Drone ip:{ip}\n")

        await asyncio.sleep(3.0)

if __name__ == "__main__":
    asyncio.run(run())