import pyrealsense2 as rs
import cv2
import numpy as np
import threading

class CameraReceiver:
    def __init__(self):
        self.depth = None
        self.colour = None  # Will store either BGR or Infrared frames
        self.is_infrared = False  # Track sensor mode for later processing
        self.lock = threading.Lock()

        # Create the pipeline and config instances (DO NOT start the pipeline yet!)
        self.pipeline = rs.pipeline()
        config = rs.config()

        # Step 1: Configure the baseline depth stream
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Step 2: Adaptive stream configuration fallback
        try:
            print("Attempting to initialize Color camera stream...")
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            # Start the pipeline once with the completed configuration
            profile = self.pipeline.start(config)
            print("Success: Color and Depth streams started.")
            align_target = rs.stream.color
            
        except RuntimeError as e:
            print(f"Color camera failed or not present: {e}")
            print("Switching to Infrared camera configuration...")
            
            # Re-instantiate config to completely erase the failed color request
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            
            try:
                # Start the pipeline using the infrared config fallback
                profile = self.pipeline.start(config)
                self.is_infrared = True
                align_target = rs.stream.infrared
                print("Success: Infrared and Depth streams started.")
            except RuntimeError as ir_error:
                print(f"Critical Error: Could not open Color OR Infrared streams: {ir_error}")
                raise SystemExit("No compatible camera streams available.")

        # Step 3: Extract intrinsics based on active stream type
        active_stream_type = rs.stream.infrared if self.is_infrared else rs.stream.color
        active_stream = profile.get_stream(active_stream_type)
        intrinsics = active_stream.as_video_stream_profile().get_intrinsics()

        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.cx = intrinsics.ppx
        self.cy = intrinsics.ppy
        self.height = intrinsics.height
        self.width = intrinsics.width

        # Step 4: Extract depth scale and set dynamic frame aligner
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(align_target)

        self.frames_aligned = True
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        try:
            while self.running:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                aligned_frames = self.align.process(frames)
                depth_frame = aligned_frames.get_depth_frame()
                
                # Dynamically fetch the matching target frame
                if self.is_infrared:
                    color_frame = aligned_frames.get_infrared_frame(1)
                else:
                    color_frame = aligned_frames.get_color_frame()

                if depth_frame and color_frame:
                    with self.lock:
                        self.colour = np.asanyarray(color_frame.get_data()).copy()
                        self.depth = np.asanyarray(depth_frame.get_data()).copy()
                    print("frame received")

                # Explicitly clear variables for memory management on Orange Pi
                frames = None
                aligned_frames = None
                depth_frame = None
                color_frame = None

        finally:
            try:
                self.pipeline.stop()
                print("Pipeline stopped cleanly inside thread loop.")
            except Exception as e:
                print(f"Error while stopping pipeline: {e}")

    def get_depth_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy()
        
    def get_RGB_frame(self):
        with self.lock:
            return None if self.colour is None else self.colour.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=6.0)
        if self.thread.is_alive():
            print("Warning: capture thread did not stop cleanly")
