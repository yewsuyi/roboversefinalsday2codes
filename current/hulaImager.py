import cv2
import numpy as np

def detect_aruco_markers(image, dictionary_type=cv2.aruco.DICT_7X7_1000):
    """
    Detects ArUco markers in a BGR or RGB image.
    
    Parameters:
    - image: numpy.ndarray, the BGR or RGB image frame.
    - dictionary_type: cv2.aruco.Dict, the dictionary of the target marker.
                       Defaults to cv2.aruco.DICT_5X5_250.
                       
    Returns:
    - detected_ids: list of detected marker IDs (or an empty list if none).
    - marked_image: image with bounding boxes drawn around the markers.
    """
    # 1. Convert BGR to Grayscale (ArUco detection requires grayscale)
    if len(image.shape) == 3:  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # 2. Set up the ArUco detector parameters and dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_type)
    parameters = cv2.aruco.DetectorParameters()
    
    # 3. Detect the markers
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    # Create a copy of the image to draw bounding boxes
    marked_image = image.copy()
    detected_ids = []
    
    # 4. Process results if any markers are found
    if ids is not None:
        detected_ids = ids.flatten().tolist()
        # Draw boundaries and IDs on the image
        cv2.aruco.drawDetectedMarkers(marked_image, corners, ids)
        print(detected_ids)
        
    return detected_ids, marked_image
