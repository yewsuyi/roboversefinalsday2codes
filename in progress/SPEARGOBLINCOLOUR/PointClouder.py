import numpy as np
import asyncio
# from Ywidrone import Drone
# from top_down import depth_to_xy_map
from customtopdown import depth_to_xy_map, local_to_ned_global, depth_to_xy_map_with_pitchdown
from ScanMap import ScanMapper


class GlobalMapper:

    def __init__(
        self,
        cam_height=1.0,
        obs_h_min=0.1,
        obs_h_max=1.5,
        depth_min=0.2,
        depth_max=15.0,
        use_pitchdown=False,
        pitchdown=0.0,
        yaw_in_degrees=False,
        yaw_clockwise=True,
        yaw_smoothing=0.7,
        map_yaw_error_rad=-0.09,
        scans_per_pointcloud=4
        
    ):
        # self.K = np.array([ #for gazebo depth cam
        #     [433.0, 0.0, 320.0],
        #     [0.0, 433.0, 240.0],
        #     [0.0, 0.0, 1.0],
        # ], dtype=np.float32)

        self.cam_height = cam_height
        self.obs_h_min = obs_h_min
        self.obs_h_max = obs_h_max
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.use_pitchdown = use_pitchdown
        self.pitchdown = pitchdown

        self.yaw_in_degrees = yaw_in_degrees
        self.yaw_clockwise = yaw_clockwise
        self.yaw_smoothing = yaw_smoothing
        self.map_yaw_error_rad = map_yaw_error_rad

        #NEW
        self.scancount:int = 0
        self.scans_per_pointcloud = scans_per_pointcloud

        self.last_yaw_rad = 0.0
        self.first_frame = True

        # # accumulated global obstacle points in [north, east]
        # self.global_points = np.empty((0, 2), dtype=np.float32)

    #from GlobalMapperV2
    def _local_to_ned_global(self, local_xy, north, east, yaw_rad):
        # print(f"post-depth-img coords: {north} | {east} | {yaw_rad}")

        """Transform local (X_cam=right, Z_cam=forward) to NED global (north, east)"""
        X_cam = local_xy[:, 0]
        Z_cam = local_xy[:, 1]
        X = Z_cam
        Y = X_cam

        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
       
        X_new = X * c - Y * s
        Y_new = X * s + Y * c
        return np.column_stack([north+X_new, east+Y_new])

    #from GlobalMapperV2
    def update_frame(self, depth_img, state, scanmapper, receiver, north, east, down, yaw_deg, yaw_rad):

        # print("\n---update-frame---")

        yaw_rad = np.arctan2(np.sin(yaw_rad),np.cos(yaw_rad)) #clamps angle btwn -pi,pi

        if yaw_rad is None:
            print("Warning: yaw is None, using 0.0")
            yaw_rad = 0.0
        if not self.yaw_clockwise: yaw_rad = -yaw_rad # convert yaw

        # yaw = self.smooth_yaw() #check below
        yaw = yaw_rad

        # extract local obstacle coordinates
        # depth_to_xy_map_with_pitchdown
        # depth_to_xy_map

        if self.use_pitchdown:
            obs_Xcoords, obs_Zcoords = depth_to_xy_map_with_pitchdown(
                depth_img,
                receiver.height, receiver.width, 
                receiver.fx, receiver.fy, receiver.cx, receiver.cy,
                receiver.depth_scale,
                # dronecam_height=self.cam_height,
                dronecam_height= -down,
                obs_h_min=self.obs_h_min,
                obs_h_max=self.obs_h_max,
                MIN_DEPTH=self.depth_min,
                MAX_DEPTH=self.depth_max,
                pitchdown=self.pitchdown
            )
        else:
            obs_Xcoords, obs_Zcoords = depth_to_xy_map(
                depth_img,
                receiver.height, receiver.width, 
                receiver.fx, receiver.fy, receiver.cx, receiver.cy,
                receiver.depth_scale,
                # dronecam_height=self.cam_height,
                dronecam_height= -down,
                obs_h_min=self.obs_h_min,
                obs_h_max=self.obs_h_max,
                MIN_DEPTH=self.depth_min,
                MAX_DEPTH=self.depth_max,
            )

        if (obs_Xcoords.size == 0) or (obs_Zcoords.size == 0): return False

        # transform and accumulate
        obs_N_m, obs_E_m = local_to_ned_global(obs_Xcoords, obs_Zcoords, north, east, yaw)
        

        scanmapper.blockoff_scanmap(obs_N_m, obs_E_m)
        scanmapper.imprint(east, north, yaw_deg)
        # if not scanmapper.imprint(east, north, yaw_deg):

            #if imprinting fails, tell the pilot
            # state.double_scanned_flag = True
        


        # self.global_points = np.vstack([self.global_points, global_pts]) #TEST NOT SAVING THIS - NOT USED ANYWHERE
        # global_pts = self._local_to_ned_global(xy_obstacles, north, east, yaw)
        # self.global_points = np.vstack([self.global_points, global_pts.astype(np.float32)])
        return True



    # def get_global_points(self):
    #     return self.global_points.copy()

    # def save_points(self, filename="global_obstacles.npy"):
    #     np.save(filename, self.global_points)
    #     print(f"Saved {len(self.global_points)} points to {filename}")

    def smooth_yaw(self):
        # NEW smoothing
        # if self.first_frame:
        #     self.last_yaw_rad = yaw_rad
        #     self.first_frame = False
        # else:
        #     yaw_rad = self.yaw_smoothing * yaw_rad + (1 - self.yaw_smoothing) * self.last_yaw_rad
        # self.last_yaw_rad = yaw_rad
        # NEW smoothing

        # OLD smoothing
        # # simple exponential smoothing
        # if self.first_frame:
        #     self.last_yaw_rad = yaw_rad
        #     self.first_frame = False
        # else:
        #    yaw_rad = self.yaw_smoothing * yaw_rad + (1.0 - self.yaw_smoothing) * self.last_yaw_rad
        #     self.last_yaw_rad = yaw_rad
        # OLD smoothing
        return


async def collect_and_map_frame(receiver, mapper:GlobalMapper, state, scanmapper, manual_override=False, parser=None, use_uwb_mode=True):

    depth_img = receiver.get_depth_frame()
    if depth_img is None: # no depth data received
    	#print("No depth frame received, skipping this cycle.")
    	return
    # np.nan_to_num(depth_img, copy=False, nan=20.0, posinf=20.0) #NEW TEST

    #COLLISION WARNING
    leftspace, midspace, rightspace = await compute_clearance(depth_img)

    #UPDATE leftspace etc. in STATE
    state.leftspace = leftspace
    state.midspace = midspace
    state.rightspace = rightspace
    mapper.scancount += 1

    #UPDATE POINTCLOUD AND IMPRINT
    if manual_override:
        updated = mapper.update_frame(
            depth_img, state, scanmapper, receiver,
            state.north, state.east, state.down, state.yaw_deg, state.yaw_rad
        )
        return leftspace, midspace, rightspace, updated
    


    # else: #DO PROCKING
    #     if leftspace > 8.0 and not state.left_freelook_proced:
    #         state.left_freelook_proced = True
    #         # print("LEFT FREELOOK PROCED")
    #     if rightspace > 8.0 and not state.right_freelook_proced:
    #         state.right_freelook_proced = True
    #         # print("RIGHT FREELOOK PROCED")
    #     #NEW
    #     state.left_freelook_proced = False
    #     state.right_freelook_proced = False
    #     #NEW
    

    
    if not state.blockpointclouding:
        if mapper.scancount >= mapper.scans_per_pointcloud:
            mapper.scancount = 0

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

            updated = mapper.update_frame(
                depth_img, state, scanmapper, receiver,
                north, east, state.down, state.yaw_deg, state.yaw_rad
            )
            return leftspace, midspace, rightspace, updated


    return leftspace, midspace, rightspace, False #updated is just true or false


async def scannow(receiver, mapper, state, scanmapper, update_pointcloud:bool):

    data = await collect_and_map_frame(receiver, mapper, state, scanmapper, update_pointcloud)
    if not data: print("--- no depthcam data ---")
    else:
        # leftspace, midspace, rightspace, mapupdated = data
        pass
        # if not mapupdated: print("--- pointclouding blocked / failed ---")

    return data



async def depthcam_pointcloud_task(
        drone, 
        receiver,
        mapper: GlobalMapper,
        state,
        scanmapper: ScanMapper,
        stop_event: asyncio.Event,
        loopdelay: float = 1.0,
        parser=None, use_uwb_mode=True,
):
    """
    Background task updating the depth camera pointcloud concurrently.
    """
    print("Depth camera pointcloud task started...")

    try:
        while True:
            if stop_event.is_set(): break
            # if state.blockpointclouding: #moved to inside collect_and_map_frame
            #     # print("Depthscan blocked") #TODO add automatic blocking if current pose deviates from prev pose by too much
            #     await asyncio.sleep(loopdelay)
            #     continue

            data = await collect_and_map_frame(receiver, mapper, state, scanmapper,  parser, use_uwb_mode)
            if not data: print("--- no depthcam data ---")
            else:
                # leftspace, midspace, rightspace, mapupdated = data
                pass
                # if not mapupdated: print("--- pointclouding blocked /failed ---")

            await asyncio.sleep(loopdelay)

    except asyncio.CancelledError: print("Pointcloud updater task cancelled.")
    except Exception as e: print(f"Pointcloud updater error: {type(e).__name__}: {e}")

#COMPUTES CLEARANCE ON EACH SIDE
async def compute_clearance(depth_map, min_threshold=0.01):
        h, w = depth_map.shape
        # print("global min:", np.nanmin(depth_map))
        # print("global max:", np.nanmax(depth_map))
        # print("nan count:",    np.isnan(depth_map).sum())
        # print("inf count:",    np.isinf(depth_map).sum())
        # clean_map = depth_map.copy()
        # clean_map[ ~( np.isfinite(depth_map) & (depth_map >= min_threshold) )] = 30.0
        # clean_map = np.where(
        #     np.isfinite(depth_map) & (depth_map >= min_threshold),
        #     depth_map,
        #     np.nan
        # )
        

        clean_map = depth_map.copy()
        # clean_map[np.isposinf(depth_map)] = 30.0
        # clean_map[(np.isposinf(depth_map)) | (np.isnan(depth_map)) | (np.isneginf(depth_map)) ] = 30.0 #TRY THIS
        clean_map[np.isneginf(clean_map)] = 30.0
        # clean_map[np.isneginf(depth_map)] = 0.1
        # danger_mask = ~(np.isfinite(clean_map)) | (clean_map < min_threshold)
        # clean_map[danger_mask] = min_threshold

        left = np.nanpercentile(clean_map[h//3:2*h//3, :w//3], 20)
        center = np.nanpercentile(clean_map[h//3:2*h//3, w//3:2*w//3], 20)
        right = np.nanpercentile(clean_map[h//3:2*h//3, 2*w//3:], 20)

        if center == 1.0:
            # print("\n\nWARN!!! 1.0 for midspace received, ignoring...\n\n")
            # print(":w: center=1.0", end='')
            # print("\nnan count:",    np.isnan(depth_map).sum())
            # print("posinf count:",    np.isposinf(depth_map).sum())
            # print("neginf count:",    np.isneginf(depth_map).sum())
            # print("POST neginf count:",    np.isneginf(clean_map).sum())
            # print("0.0 count:",    np.count_nonzero(clean_map == 0.0))
            # print("<0.5 count:",    np.count_nonzero(clean_map < 0.5))
            center = 50.0
        if left == 1.0: left = 50.0
        if right == 1.0: right = 50.0
            

        return left, center, right

def get_uwb_position_NE(UWBparser):
        x, y, update_time, validity = UWBparser.get_tag_position(0)
        if x is None: return None, None, None
        
        N, E = UWBxy_to_globalNE(x, y)
        return N, E, validity