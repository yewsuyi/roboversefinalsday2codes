import cv2
import time
from dola import Dola
from pyhulax import DroneAPI
from pyhulax.video import VideoStream, VideoDisplay
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
        drone.connect(drone_ip)
        
        # 3. Create Video Stream Endpoint Pipeline
        print("[4/4] Opening video stream instance...")
        stream = drone.create_video_stream()
        drone.set_video_stream(True)
        
        if stream is not None:
            stream.start()
            print(">>> Stream Active. Display window rendering. Press 'Q' on keyboard to close.")
            
            while True:
                frame_obj = stream.latest_frame
                if frame_obj is not None:
                    # Convert internal array bytes to a standard RGB frame
                    rgb_array = frame_obj.to_rgb()
                    # Convert to BGR format which OpenCV natively displays correctly
                    bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
                    detect_aruco_markers(bgr_image)
                    
                    cv2.imshow(f"HULA Target [{plane_id}] Feed", bgr_image)
                
                # Exit loop cleanly if 'q' is pressed in the image window
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            print("[ERROR] Stream instantiation object dropped by drone firmware.")

    except Exception as error:
        print(f"[CRITICAL FAILURE] Pipeline dropped: {error}")
    finally:
        print("Safely shutting down streams and window frameworks...")
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostic()