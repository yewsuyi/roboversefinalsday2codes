# Get RGB image from Intel RealSense camera and display it using OpenCV
import asyncio

# from CameraReceivers.RealsenseCamera import CameraReceiver #DONT IMPORT HERE
import pyrealsense2 as rs
import cv2
import numpy as np
import os
from rknnlite.api import RKNNLite

from rknndecoder import decode_yolov11_rknn, draw_detections

# Create the 'images' directory if it doesn't already exist
OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARUCO_DICT = cv2.aruco.DICT_7X7_1000
# --------------------------------------------------
# CONFIG & CLASS MAPS
# --------------------------------------------------
RKNN_MODEL_PATH = "models/my_model.rknn"
MODEL_SIZE = (640, 640)

YOLO_CLASS_NAMES = {
    0: "landing_pad",      # Tailor these to match your specific model classes
}

# Initialize the ArUco dictionary, detector and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def load_rknn_model(model_path=RKNN_MODEL_PATH):
    rknn = RKNNLite()
    rknn.load_rknn(model_path)
    rknn.init_runtime()
    return rknn


async def imager_task(
    receiver,
    stop_event,
    loopdelay,
    valid_aruco_ids,
    rknn=None,
    rknn_model_path=RKNN_MODEL_PATH,
):
    if rknn is None:
        rknn = load_rknn_model(rknn_model_path)

    try:
        while not stop_event.is_set():
            # Get whatever video frame is currently streaming
            color_frame = receiver.get_RGB_frame()
            depth_frame = receiver.get_depth_frame()
            
            if color_frame is None or depth_frame is None:
                await asyncio.sleep(loopdelay)
                continue
            
            # 1. ArUco Marker Detection for Landing Pad Verification
            # ==============================================================
            should_run_rknn = False  # Flag to control whether to run YOLOv11 inference
            detected_ids, marked_image = detect_aruco_markers(color_frame)
            for id in detected_ids:
                if id in valid_aruco_ids:
                    should_run_rknn = True
                    print("Valid ID detected! Activating YOLOv11 Landing Pad Verification...")
                    break

            # 2. Landing Pad Verification with YOLOv11 on RKNN NPU
            # ============================================================
            if should_run_rknn: 
                # FIX 1: We must capture the returned image into a variable
                marked_image = detect_landing_pad(
                    color_frame,
                    marked_image,
                    rknn,
                )

            # 3. Continuous Window Video Output Render
            # FIX 2: Show "marked_image" instead of "colour_image"
            cv2.imshow("RealSense Unified Gated Pipeline", marked_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            await asyncio.sleep(loopdelay)

    except Exception as e:
        print(f"Critical execution breakdown in pipeline loop: {e}")
    finally:
        rknn.release()
        cv2.destroyAllWindows()

def detect_aruco_markers(image):
    """
    Detects ArUco markers in a BGR OpenCV image.
    
    Parameters:
    - image: numpy.ndarray, the BGR image frame.
                       
    Returns:
    - detected_ids: list of detected marker IDs (or an empty list if none).
    - marked_image: image with bounding boxes drawn around the markers.
    """
 
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
    
    # Create a copy of the image to draw bounding boxes
    marked_image = image.copy()
    detected_ids = []
    
    # 4. Process results if any markers are found
    if ids is not None:
        detected_ids = ids.flatten().tolist()
        # Draw boundaries and IDs on the image
        cv2.aruco.drawDetectedMarkers(marked_image, corners, ids)

        # Turn list [1, 2] into a string format "1_2" for the filename
        id_string = "_".join(str(x) for x in detected_ids)
        filename = f"marker_{id_string}.png"
        # Combine folder path and filename (e.g., "bean/pad_1.png")
        filepath = os.path.join(OUTPUT_DIR, filename)
        # Save the image frame to your disk
        cv2.imwrite(filepath, marked_image)


        
        
        print(detected_ids)
        
    return detected_ids, marked_image

def detect_landing_pad(color_frame, marked_image, rknn):
    # Setup dimensions for the model input
    model_size = MODEL_SIZE
    
    # Get original dimensions from the input image
    image_height, image_width = color_frame.shape[:2]
        
    # Step 1: Pre-process the frame for the model
    img_for_model = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_for_model, model_size)
    img_input = np.expand_dims(img_resized.astype(np.uint8), axis=0) # Add batch dimension (1, 640, 640, 3)

    # Step 2: Hardware NPU accelerated inference execution
    outputs = rknn.inference(inputs=[img_input])

    # Step 3: Run custom decoding script to convert raw model outputs to boxes
    final_boxes, final_scores, final_classes = decode_yolov11_rknn(
        outputs=outputs,
        img_shape=(image_height, image_width),
        model_input_size=model_size
    )
    # Step 4: Landing pad confirmation if object verified
    if len(final_boxes) > 0:
        print(f"Landing Pad Verified! NPU found {len(final_boxes)} validation target(s).")
        
        # Graph bounding frames directly on top of screen output
        marked_image = draw_detections(
            marked_image, final_boxes, final_scores, final_classes, YOLO_CLASS_NAMES
        )

        # Extract location metrics from detected targets
        for box, score, cls_id in zip(final_boxes, final_scores, final_classes):
            x1, y1, x2, y2 = box.astype(int)
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

    # Return marked_image directly (it's either modified by draw_detections or left clean)
    return marked_image
