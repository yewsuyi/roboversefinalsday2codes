# Get RGB image from Intel RealSense camera and display it using OpenCV
import asyncio

# from CameraReceivers.RealsenseCamera import CameraReceiver #DONT IMPORT HERE
import pyrealsense2 as rs
import cv2
import numpy as np
import os
import time

try:
    from rknnlite.api import RKNNLite
except ImportError:
    RKNNLite = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

from rknndecoder import decode_yolov11_rknn, draw_detections

OUTPUT_DIR = "images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ARUCO_DICT = cv2.aruco.DICT_7X7_1000
# --------------------------------------------------
# CONFIG & CLASS MAPS
# --------------------------------------------------
RKNN_MODEL_PATH = "models/my_model.rknn"
ULTRALYTICS_MODEL_PATH = "models/my_model.pt"
MODEL_SIZE = (640, 640)
ULTRALYTICS_CONF_THRES = 0.5

YOLO_CLASS_NAMES = {
    0: "Landing pad",      # Tailor these to match your specific model classes
}

# Initialize the ArUco dictionary, detector and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)


def load_rknn_model(model_path):
    if RKNNLite is None:
        raise RuntimeError("rknnlite is not installed")

    rknn = RKNNLite()
    if rknn.load_rknn(model_path) != 0:
        raise RuntimeError(f"failed to load RKNN model: {model_path}")
    if rknn.init_runtime() != 0:
        raise RuntimeError("failed to initialize RKNN runtime")
    return rknn


def load_ultralytics_model(model_path=ULTRALYTICS_MODEL_PATH):
    if YOLO is None:
        raise RuntimeError("ultralytics is not installed")

    return YOLO(model_path)


def load_detector(rknn_model_path=RKNN_MODEL_PATH, yolo_model_path=ULTRALYTICS_MODEL_PATH):
    try:
        print("Loading RKNN landing-pad detector...")
        return "rknn", load_rknn_model(rknn_model_path)
    except Exception as e:
        print(f"RKNN detector unavailable, falling back to Ultralytics YOLO: {e}")

    return "ultralytics", load_ultralytics_model(yolo_model_path)


async def imager_task(
    receiver,
    stop_event,
    loopdelay,
    valid_aruco_ids,
    detector_backend=None,
    detector=None,
    rknn_model_path=RKNN_MODEL_PATH,
    yolo_model_path=ULTRALYTICS_MODEL_PATH,
):
    if detector is None:
        detector_backend, detector = load_detector(rknn_model_path, yolo_model_path)

    try:
        while not stop_event.is_set():
            # Get whatever video frame is currently streaming
            color_frame = receiver.get_RGB_frame()
            if color_frame is None:
                await asyncio.sleep(loopdelay)
                continue
            
            # 1. Optional ArUco Marker Detection
            # ==============================================================
            detected_ids, marked_image = detect_aruco_markers(color_frame)
            for id in detected_ids:
                if id in valid_aruco_ids:
                    print("Valid ArUco ID detected.")
                    break

            # 2. Landing Pad Verification with YOLOv11
            # ============================================================
            marked_image = detect_landing_pad(
                color_frame,
                marked_image,
                detector_backend,
                detector,
            )

            # 3. Continuous Window Video Output Render
            cv2.imshow("RealSense Unified Gated Pipeline", marked_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            await asyncio.sleep(loopdelay)

    except Exception as e:
        print(f"Critical execution breakdown in pipeline loop: {e}")
    finally:
        if detector_backend == "rknn":
            detector.release()
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
        id_string = "_".join(str(x) for x in detected_ids)
        filename = f"marker_{id_string}_{int(time.time() * 1000)}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), marked_image)

        print(detected_ids)
        
    return detected_ids, marked_image

def detect_landing_pad(color_frame, marked_image, detector_backend, detector):
    # Setup dimensions for the model input
    model_size = MODEL_SIZE
    
    # Get original dimensions from the input image
    image_height, image_width = color_frame.shape[:2]
        
    if detector_backend == "rknn":
        # Step 1: Pre-process the frame for the RKNN model
        img_for_model = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_for_model, model_size)
        img_input = np.expand_dims(img_resized.astype(np.uint8), axis=0) # Add batch dimension (1, 640, 640, 3)

        # Step 2: Hardware NPU accelerated inference execution
        outputs = detector.inference(inputs=[img_input])

        # Step 3: Run custom decoding script to convert raw model outputs to boxes
        final_boxes, final_scores, final_classes = decode_yolov11_rknn(
            outputs=outputs,
            img_shape=(image_height, image_width),
            model_input_size=model_size
        )
    else:
        results = detector(color_frame, imgsz=model_size[0], conf=ULTRALYTICS_CONF_THRES, verbose=False)
        boxes = results[0].boxes
        final_boxes = boxes.xyxy.cpu().numpy()
        final_scores = boxes.conf.cpu().numpy()
        final_classes = boxes.cls.cpu().numpy().astype(int)
    # Step 4: Landing pad confirmation if object verified
    if len(final_boxes) > 0:
        print(f"Landing Pad Verified! Found {len(final_boxes)} validation target(s).")
        
        # Graph bounding frames directly on top of screen output
        marked_image = draw_detections(
            marked_image, final_boxes, final_scores, final_classes, YOLO_CLASS_NAMES
        )
        filename = f"landing_pad_{int(time.time() * 1000)}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), marked_image)

    # Return marked_image directly (it's either modified by draw_detections or left clean)
    return marked_image
