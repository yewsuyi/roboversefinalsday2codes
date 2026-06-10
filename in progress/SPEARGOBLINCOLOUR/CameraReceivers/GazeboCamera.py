from gz.transport13 import Node
from gz.msgs10.image_pb2 import Image
import numpy as np
import threading

#no need colour cam, we are not testing that

class CameraReceiver:
    def __init__(self, topic="/depth_camera"):

        self.depth_scale = 1
        self.fx = 433.0
        self.fy = 433.0
        self.cx = 320.0
        self.cy = 240.0
        # self.u_coords, self.v_coords = np.meshgrid(
        #     np.arange(WIDTH),
        #     np.arange(HEIGHT)
        # )
        self.height = None
        self.width = None

        self.node = Node()
        self.depth = None
        self.lock = threading.Lock()

        # ✅ FIXED LINE
        self.node.subscribe(Image, topic, self.callback)

    def callback(self, msg):
        depth = np.frombuffer(msg.data, dtype=np.float32)
        depth = depth.reshape((msg.height, msg.width))

        with self.lock:
            self.depth = depth

    def get_depth_frame(self):
        with self.lock:
            return None if self.depth is None else self.depth.copy()
        
    # def get_RGB_frame(self):
        
    def stop(self):
        print("gazebo cam stopped (no action needed)")
