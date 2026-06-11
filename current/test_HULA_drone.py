import cv2
import time
import asyncio
from dola import Dola
from pyhulax import DroneAPI
from pyhulax.video import VideoStream, VideoDisplay
# Import the asynchronous tracking engine you wrote previously
from hulaImager import detect_aruco_markers

def run_diagnostic():
    # 1. Initialize Network Discovery
    explorer = Dola()
    explorer.start()
    print("[1/4] Scanning Wi-Fi network for active HULA drones...")
    
    # Listen on network interfaces for 5 seconds
    devices = explorer.get_all_ips(listen_seconds=5)
    explorer.stop()

    if not devices:
        print("[ERROR] No HULA drones discovered! Double-check your PC is connected to the drone's Wi-Fi network.")
        return

    # Extract the first identified aircraft mapping
    plane_id, drone_ip = list(devices.items())[0]
    print(f"[2/4] Discovered Drone ID: {plane_id} on Network IP: {drone_ip}")

    # 2. Establish Control API Client Link
    drone = DroneAPI()
    try:
        print(f"[3/4] Initializing connection to {drone_ip}...")
        # Direct connection to the parsed IP address string
        drone.connect(str(drone_ip))
        
        # 3. Fire up the Async Tracking Engine
        print("[4/4] Passing drone object to the asynchronous ArUco processing engine...")
        
        # Since your detect_aruco_markers takes (id, drone) and manages its own loop/stream,
        # we run it directly inside Python's asyncio event loop runner.
        asyncio.run(detect_aruco_markers(id=plane_id, drone=drone, loopdelay=0.01))

    except Exception as error:
        print(f"[CRITICAL FAILURE] Pipeline dropped: {error}")
    finally:
        print("Safely shutting down streams and window frameworks...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostic()
