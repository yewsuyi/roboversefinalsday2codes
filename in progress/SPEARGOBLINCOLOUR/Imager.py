# Get RGB image from Intel RealSense camera and display it using OpenCV
import asyncio

# from CameraReceivers.RealsenseCamera import CameraReceiver #DONT IMPORT HERE
import pyrealsense2 as rs
import cv2
from UWBaller import UWBxy_to_globalNE
from customtopdown import depth_pixel_to_xz
import numpy as np

ARUCO_DICT = cv2.aruco.DICT_7X7_1000
# --------------------------------------------------
# CONFIG & CLASS MAPS
# --------------------------------------------------
VALID_IDS = {1, 2, 3, 42}  # Example set of valid ArUco landing pad/marker IDs

YOLO_CLASS_NAMES = {
    0: "landing_pad",      # Tailor these to match your specific model classes
}

# Initialize the ArUco dictionary, detector and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


async def imager_task(
    receiver, mapper, state, parser, scanmapper, 
    use_uwb_mode, use_pitchdown, pitchdown, stop_event, loopdelay,
    valid_aruco_ids,
    use_infrared=True  # Defaulting to True for your setup
):
    try:
        while not stop_event.is_set():
            
            RGB_image = receiver.get_RGB_frame()
            marked_image = RGB_image.copy()
            grayscale_image = cv2.cvtColor(RGB_image, cv2.COLOR_BGR2GRAY)

            corners, ids, rejected = detector.detectMarkers(grayscale_image)
            
            
            
            if ids is not None: 
                print('fehwf',ids)
                cv2.aruco.drawDetectedMarkers(marked_image, corners, ids)
            
            # ============================================================
            # STEP 5: VISUAL SCREEN LAYOUT RENDER OUTPUTS
            # ============================================================
            # Display 'colour_image' (which holds the drawn markers/detections on Left IR) 
            # alongside the raw Right IR stream
            cv2.imshow("grayimg", marked_image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            await asyncio.sleep(loopdelay)

    except Exception as e:
        print(f"Critical execution breakdown in pipeline loop: {e}")
    finally:
        cv2.destroyAllWindows()


def get_uwb_position_NE(UWBparser, uwb_tag):
        x, y, update_time, validity = UWBparser.get_tag_position(uwb_tag)
        if x is None: return None, None, None
        
        N, E = UWBxy_to_globalNE(x, y)
        return N, E, validity

async def aruco_detected(
    receiver,mapper,state,parser,
    use_uwb_mode, use_pitchdown, pitchdown,
    corresp_depth_image,
    detected_pixel_x, detected_pixel_y, #TODO FEED THIS
    colour_image_height, colour_image_width, #TODO FEED THIS
    ids, valid_aruco_ids
):
    obs_N_m, obs_E_m = await pinpoint_spot(
        receiver,mapper,state,parser,
        use_uwb_mode, use_pitchdown, pitchdown,
        corresp_depth_image,
        detected_pixel_x, detected_pixel_y, #TODO FEED THIS
        colour_image_height, colour_image_width) #TODO FEED THIS

    if obs_N_m is not None and obs_E_m is not None:
        print("ALERT!!! pinpoint_spot() returned NONE")
        return False #pinpoint got issue

async def landingpad_detected(): pass
        
async def zone_detected(): pass # Get RGB image from Intel RealSense camera and display it using OpenCV
