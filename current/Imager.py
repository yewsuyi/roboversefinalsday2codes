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
