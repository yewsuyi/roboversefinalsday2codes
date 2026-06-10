# Same as RealsenseCameraALIGNED.py but depth frame might have a different perspective as colour frame (parallax error)

import pyrealsense2 as rs
import cv2
import numpy as np
import threading

class CameraReceiver:
    def __init__(self):
        self.depth = None
        self.colour = None
        self.lock = threading.Lock()

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)

        depth_stream = profile.get_stream(rs.stream.depth)
        depth_intrinsics = (depth_stream.as_video_stream_profile().get_intrinsics())

        self.fx = depth_intrinsics.fx
        self.fy = depth_intrinsics.fy
        self.cx = depth_intrinsics.ppx
        self.cy = depth_intrinsics.ppy
        self.height = depth_intrinsics.height
        self.width = depth_intrinsics.width

        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        self.frames_aligned = False
        # Start background thread to continuously grab frames
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        try:
            while self.running:

                # frames = self.pipeline.wait_for_frames()
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                # aligned_frames = self.align.process(frames)
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if color_frame:
                    with self.lock: self.colour = np.asanyarray(color_frame.get_data()).copy()
                        
                if depth_frame:
                    with self.lock: self.depth = np.asanyarray(depth_frame.get_data()).copy()

                frames = None
                depth_frame = None
                color_frame = None

        finally:
            self.pipeline.stop()

    def get_depth_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy() # might have arrary with np.asanyarry
        
    def get_RGB_frame(self):
        with self.lock:
            return None if self.colour is None else self.colour.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=6.0)  # slightly longer than timeout_ms=5000
        if self.thread.is_alive():
            print("Warning: capture thread did not stop cleanly")