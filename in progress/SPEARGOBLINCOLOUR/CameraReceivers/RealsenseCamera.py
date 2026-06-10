import pyrealsense2 as rs
import cv2
import numpy as np
import threading
import time

class CameraReceiver:
    def __init__(self):
        self.depth = None
        self.colour = None
        self.is_infrared = False  # Tracks if we fall back to infrared
        self.lock = threading.Lock()

        self.pipeline = rs.pipeline()
        config = rs.config()
        
        # Step 1: Base configuration (Depth is mandatory)
        config.enable_stream(rs.stream.depth, 320, 240, rs.format.z16, 30)

        # Step 2: Adaptive stream resolution strategy
        try:
            print("Attempting to initialize Color camera stream (YUYV fallback optimized)...")
            # Using YUYV format uses 50% less USB bandwidth than BGR8, bypassing Rockchip port limitations
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            # config.enable_stream(rs.stream.color, 640, 480, rs.format.yuyv, 30)
            profile = self.pipeline.start(config)
            print("🚀 Success: Started with Color + Depth configuration.")
        except RuntimeError as e:
            print(f"⚠️ Color camera stream failed to resolve: {e}")
            print("🔄 Falling back strictly to Infrared + Depth configuration...")
            
            # Re-initialize config to drop the broken color stream request completely
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            # config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
            #PRINTS()
            config.enNO Dable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            
            try:
                profile = self.pipeline.start(config)
                self.is_infrared = True
                print("🚀 Success: Started with Infrared + Depth configuration.")
            except RuntimeError as critical_err:
                print(f"❌ Critical Error: Could not resolve ANY camera streams: {critical_err}")
                raise SystemExit("Camera hardware unresolvable.")

        # 🚀 THE CRITICAL FIX: Give the hardware sensors time to boot and warm up
        #print("⏳ Warming up camera hardware sensors for 2 seconds...")
        #time.sleep(2)

        # Step 3: Extract depth stream intrinsics
        depth_stream = profile.get_stream(rs.stream.depth)
        depth_intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        self.fx = depth_intrinsics.fx
        self.fy = depth_intrinsics.fy
        self.cx = depth_intrinsics.ppx
        self.cy = depth_intrinsics.ppy
        self.height = depth_intrinsics.height
        self.width = depth_intrinsics.width

        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

        self.frames_aligned = False
        self.running = True
        print("\nSTARTING THREAD...\n")
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        try:
            while self.running:
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                depth_frame = frames.get_depth_frame()
                #print(f"depth frame received: depth_frame is {depth_frame is None}")
                # Fetch color or infrared based on active streaming strategy
                if self.is_infrared:
                    raw_frame = frames.get_infrared_frame(1)
                else:
                    raw_frame = frames.get_color_frame()

                # Validate that BOTH frames actually contain valid spatial matrices
                if depth_frame and raw_frame:
                    #print("both frames ready")
                    img_data = np.asanyarray(raw_frame.get_data()).copy()
                    depth_data = np.asanyarray(depth_frame.get_data()).copy()
                    
                    # Ensure array allocations contain structural data before committing
                    if np.any(img_data) and np.any(depth_data):
                        # Convert 1-channel Infrared to 3-channel fake BGR
                        if self.is_infrared:
                            img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                        # else:
                            # Convert YUYV format to standard BGR for OpenCV display outputs
                            # img_data = cv2.cvtColor(img_data, cv2.COLOR_YUV2BGR_YUYV)
                            # img_data = img_data

                        with self.lock:
                            #print("assigning actual frames to memory...")
                            self.colour = img_data
                            self.depth = depth_data

                frames = None
                depth_frame = None
                raw_frame = None

        except RuntimeError as err:
            print(f"⚠️ Capture Loop Interrupted: {err}")
        finally:
            try:
                self.pipeline.stop()
                print("Pipeline stopped cleanly inside thread loop.")
            except:
                pass

    def get_depth_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy()

    def get_video_frame(self):
        return self.get_RGB_frame()
        
    def get_RGB_frame(self):
        with self.lock:
            return None if self.colour is None else self.colour.copy()

    def stop(self):
        self.running = False
        self.thread.join(timeout=6.0)
