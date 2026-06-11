import cv2
import numpy as np
import os
import asyncio

# Set up the ArUco detector parameters and dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_7X7_1000)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)       

#add id parameter in the async task, use it in naming image files for uniqueness
async def detect_aruco_markers(id, drone, loopdelay=0.1):
    """
    Detects ArUco markers in a BGR or RGB image.
    
    Parameters:
    - id: int, unique identifier for the drone or session, used for naming output files.
    - image: numpy.ndarray, the BGR or RGB image frame.
    - dictionary_type: cv2.aruco.Dict, the dictionary of the target marker.
                       Defaults to cv2.aruco.DICT_5X5_250.
                       
    Returns:
    - detected_ids: list of detected marker IDs (or an empty list if none).
    - marked_image: image with bounding boxes drawn around the markers.
    """

    drone = drone.drone #I ADDED THIS

    stream = hula_Imager_init(id, drone)
    while True: 
        
        if hula_Imager_loop(id, stream): break
        await asyncio.sleep(loopdelay)
        # Exit loop cleanly if 'q' is pressed in the image window
        
            

def hula_Imager_init(id, drone): 

    stream = drone.create_video_stream()
    drone.set_video_stream(True)
        
    if stream is not None:
        stream.start()
        print(f"Stream {id}Active. Display window rendering. Press 'Q' on keyboard to close.")
        return stream
    else:
        print(f"[ERROR] Stream {id} instantiation object dropped by drone firmware.")
    

def hula_Imager_loop(id, stream):
    frame_obj = stream.latest_frame
    if frame_obj is not None:

        # Convert internal array bytes to a standard RGB frame
        rgb_array = frame_obj.to_rgb()
        # Convert to BGR format which OpenCV natively displays correctly
        image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = detector.detectMarkers(gray)

        # draw the bounding boxes 
        marked_image = image.copy()
        if ids is not None:
            detected_ids = ids.flatten().tolist()
            cv2.aruco.drawDetectedMarkers(marked_image, corners, ids)
            # Turn list [1, 2] into a string format "1_2" for the filename
            id_string = "_".join(str(x) for x in detected_ids)
            filename = f"Drone_{id}_{id_string}.png"

            filepath = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(filepath, marked_image)

        cv2.imshow(f"Drone {id} Feed", marked_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): return True
    return False


    
