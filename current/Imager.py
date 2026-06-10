# Get RGB image from Intel RealSense camera and display it using OpenCV
import asyncio

# from CameraReceivers.RealsenseCamera import CameraReceiver #DONT IMPORT HERE
import pyrealsense2 as rs
import cv2
from UWBaller import UWBxy_to_globalNE
from customtopdown import depth_pixel_to_xz
import numpy as np
from customtopdown import depth_pixel_to_xz_downward

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

async def pinpoint_spot(
    receiver, mapper, state, parser,
    use_uwb_mode, use_pitchdown, pitchdown,
    depth_img, detected_pixel_x, detected_pixel_y,
    colour_image_height, colour_image_width,
):
    """
    Returns obs_N_m, obs_E_m, the global coords of the detected pixel.
    - use obs_X_u, obs_Y_u = scanmapper.worldNE_to_scanmapXY(obs_N_m, obs_E_m)
    - then set scanmapper.scanmap[obs_Y_u, obs_X_u] = 6 for VALID, 9 for INVALID
    """

    if use_uwb_mode:
        north, east, valid = get_uwb_position_NE(parser)

        if valid is None:
            print(f"UWB data MISSING, defaulting to state coords: {state.north}N | {state.east}E")
            north = state.north
            east = state.east

        elif not valid: print(f"UWB data OUTDATED, using anyway: {north}N | {east}E")

    else:
        north = state.north
        east = state.east

    if receiver.frames_aligned:
        corresp_dpixel_y = detected_pixel_y
        corresp_dpixel_x = detected_pixel_x
    else:
        c_height, c_width = colour_image_height, colour_image_width
        relative_pixel_y = detected_pixel_y/c_height
        relative_pixel_x = detected_pixel_x/c_width

        d_height = receiver.height
        d_width = receiver.width
        corresp_dpixel_y = round(relative_pixel_y * d_height)
        corresp_dpixel_x = round(relative_pixel_x * d_width)

        obsx, obsz = depth_pixel_to_xz_downward(
            depth_img,
            u=corresp_dpixel_x, #column - x
            v=corresp_dpixel_y, #row - y
            fx=receiver.fx,
            fy=receiver.fy,
            cx=receiver.cx,
            cy=receiver.cy,
            depth_scale=receiver.depth_scale,
        )
        if (obsx is None) or (obsz is None):
            print("ALERT!!!!! corresponding detection error in depth_pixel_to_xz()")
            return None, None
        
        yaw_rad = np.arctan2(np.sin(state.yaw_rad),np.cos(state.yaw_rad)) #clamps angle btwn -pi,pi
        obs_N_m, obs_E_m = mapper.local_to_ned_global(obsx, obsz, north, east, yaw_rad)
        return obs_N_m, obs_E_m


async def imager_task(
    receiver, mapper, state, parser, scanmapper, 
    use_uwb_mode, use_pitchdown, pitchdown, stop_event, loopdelay,
    valid_aruco_ids,
    use_infrared=False
):
    try:
        while not stop_event.is_set():
            # Get whatever video frame is currently streaming
            image = receiver.get_video_frame()
            corresp_depth_image = receiver.get_depth_frame()
            
            if image is None or corresp_depth_image is None:
                await asyncio.sleep(loopdelay)
                continue

            colour_image_height, colour_image_width = image.shape[:2]

            # ------------------------------------------------------------
            # DYNAMIC STREAM DETECTION BASED ON NUMPY CHANNELS
            # ------------------------------------------------------------
            if len(image.shape) == 2:
                # If shape is strictly 2D (height, width), it is INFRARED grayscale
                colour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                gray = image.copy()
            else:
                # If shape has 3 channels (height, width, 3), it is native BGR COLOR
                colour_image = image.copy()
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


            # ============================================================
            # STEP 2: RUN ARUCO DETECTION FIRST
            # ============================================================
            corners, ids, rejected = detector.detectMarkers(gray)
            
            # Initialization gate flag for RKNN execution
            should_run_rknn = False

            if ids is not None: 
                flat_ids = ids.flatten()
                print("Detected ArUco markers:", flat_ids.tolist())
                cv2.aruco.drawDetectedMarkers(colour_image, corners, ids)
                
                for marker_corners, marker_id in zip(corners, flat_ids):
                    # Check if the observed ID exists inside your VALID_IDS filter rule
                    if marker_id in VALID_IDS:
                        should_run_rknn = True  # Raise gate flag to trigger RKNN verification
                        
                    points = marker_corners.reshape((4, 2))
                    # pixelcoord_x = int(np.mean(points[:, 0]))
                    # pixelcoord_y = int(np.mean(points[:, 1]))

                    # 📍 PLACE 1: Call aruco_detected here immediately upon finding ANY ArUco marker
                    # await aruco_detected(colour_image, pixelcoord_x, pixelcoord_y, int(marker_id))

            #RUN YOLO DETECTOR HERE TODO @YANGWEI, then call this
            # await landingpad_detected(colour_image, pixelcoord_x, pixelcoord_y)
            

                    # # Map ArUco Marker location onto 3D Space coordinate matrices
                    # obs_N, obs_E = await pinpoint_spot(
                    #     receiver, mapper, state, parser,
                    #     use_uwb_mode, use_pitchdown, pitchdown,
                    #     corresp_depth_image, pixelcoord_x, pixelcoord_y,
                    #     colour_image_height, colour_image_width
                    # )

                    # if obs_N is not None and obs_E is not None:
                    #     obs_X_u, obs_Y_u = scanmapper.worldNE_to_scanmapXY(obs_N, obs_E)
                    #     scanmapper.scanmap[obs_Y_u, obs_X_u] = 6 

            # ============================================================
            # STEP 3: GATED RKNN INFERENCE (RUNS ONLY IF ID IS VALID)
            # ============================================================
            if should_run_rknn:
                print("Valid ID detected! Activating YOLOv11 Landing Pad Verification...")
                
                model_size = (640, 640)
                
                if use_infrared and len(image.shape) == 2:
                    img_for_model = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                else:
                    img_for_model = image.copy() 
                    
                img_resized = cv2.resize(img_for_model, model_size)
                img_input = np.expand_dims(img_resized, axis=0)

                # Hardware NPU accelerated inference execution
                outputs = rknn.inference(inputs=[img_input])

                # Feed tensor arrays into your custom decoding script
                final_boxes, final_scores, final_classes = decode_yolov11_rknn(
                    outputs=outputs,
                    img_shape=(colour_image_height, colour_image_width),
                    model_input_size=model_size
                )

                # ============================================================
                # STEP 4: LANDING PAD CONFIRMATION IF OBJECT VERIFIED
                # ============================================================
                if len(final_boxes) > 0:
                    print(f"Landing Pad Verified! NPU found {len(final_boxes)} validation target(s).")
                    
                    # Graph bounding frames directly on top of screen output
                    colour_image = draw_detections(
                        colour_image, final_boxes, final_scores, final_classes, YOLO_CLASS_NAMES
                    )

                    # Extract location metrics from detected targets
                    for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
                        x1, y1, x2, y2 = box.astype(int)
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # 📍 PLACE 2: Call landingzone_detected here once the landing pad is verified by RKNN
                        # await landingzone_detected(colour_image, center_x, center_y)

                        # Parse localized data frames through 3D positioning solver
                        # obs_N, obs_E = await pinpoint_spot(
                        #     receiver, mapper, state, parser,
                        #     use_uwb_mode, use_pitchdown, pitchdown,
                        #     corresp_depth_image, center_x, center_y,
                        #     colour_image_height, colour_image_width
                        # )

                        # if obs_N is not None and obs_E is not None:
                        #     obs_X_u, obs_Y_u = scanmapper.worldNE_to_scanmapXY(obs_N, obs_E)
                        #     scanmapper.scanmap[obs_Y_u, obs_X_u] = 6 

            # 5. Continuous Window Video Output Render
            cv2.imshow("RealSense Unified Gated Pipeline", colour_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            await asyncio.sleep(loopdelay)

    except Exception as e:
        print(f"Critical execution breakdown in pipeline loop: {e}")
    finally:
        cv2.destroyAllWindows()

def get_uwb_position_NE(UWBparser, uwb_tag=0):
        x, y, update_time, validity = UWBparser.get_tag_position(uwb_tag)
        if x is None: return None, None, None
        
        N, E = UWBxy_to_globalNE(x, y)
        return N, E, validity

# async def aruco_detected(
#     receiver,mapper,state,parser,
#     use_uwb_mode, use_pitchdown, pitchdown,
#     corresp_depth_image,
#     detected_pixel_x, detected_pixel_y, #TODO FEED THIS
#     colour_image_height, colour_image_width, #TODO FEED THIS
#     ids, valid_aruco_ids
# ):
#     obs_N_m, obs_E_m = await pinpoint_spot(
#         receiver,mapper,state,parser,
#         use_uwb_mode, use_pitchdown, pitchdown,
#         corresp_depth_image,
#         detected_pixel_x, detected_pixel_y, #TODO FEED THIS
#         colour_image_height, colour_image_width, #TODO FEED THIS

#     if obs_N_m is not None and obs_E_m is not None:
#         print("ALERT!!! pinpoint_spot() returned NONE"))
#         return False #pinpoint got issue

# async def landingpad_detected(): pass
