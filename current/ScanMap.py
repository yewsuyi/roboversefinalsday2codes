import numpy as np
import asyncio
# from scipy.ndimage import binary_dilation
from Astar import astar, simplifypath, pathfind
from Frontier import find_frontier_cells, find_frontier_clusters, rank_frontier_clusters, closest_cell_to_centroid

#init scanmap: scanmap = np.full((height_NORTHLENGTH, width_EASTLENGTH), 0, dtype=np.int8)

class ScanMapper:
    def __init__(self, heightcells_NORTHLENGTH, widthcells_EASTLENGTH, metrespercell, ScanmapOriginOffset_Exm, ScanmapOriginOffset_Nym, OBSTACLEMAP, scanradius, scanwidth=4.0, footprintradius=0.7, borderwallthickness=2):
        self.height = heightcells_NORTHLENGTH
        self.width = widthcells_EASTLENGTH
        self.resolution = metrespercell

        self.ScanmapOriginOffset_Exm = ScanmapOriginOffset_Exm #UWB Exm pos of the cage origin
        self.ScanmapOriginOffset_Nym = ScanmapOriginOffset_Nym #UWB Nym pos of the cage origin

        self.scanradius = scanradius
        self.scanwidth = scanwidth

        self.BORDERWALL = borderwallthickness #border thickness

        if OBSTACLEMAP is not None:
            self.scanmap = OBSTACLEMAP
            self.scanmap[self.scanmap==1] = 2 #set obstacles cells to 2
            self.scanmap[self.scanmap==0] = 1 #set free cells to 1

        else:
            self.scanmap = np.full((heightcells_NORTHLENGTH, widthcells_EASTLENGTH), 0, dtype=np.int8)
            # 2. Set the top and bottom rows to 3
            self.scanmap[0:self.BORDERWALL, :] = 2   # Top row
            self.scanmap[-self.BORDERWALL:, :] = 2  # Bottom row
            self.scanmap[:, 0:self.BORDERWALL] = 2   # Left column
            self.scanmap[:, -self.BORDERWALL:] = 2  # Right column

    def worldNE_to_scanmapXY(self, N, E, clamp=True):
        SMEx = round( (E - self.ScanmapOriginOffset_Exm)/self.resolution)
        SMNy = round( (N - self.ScanmapOriginOffset_Nym)/self.resolution)
        return (
            max(0, min(SMEx, self.width-1) ), 
            max(0, min(SMNy, self.height-1) )
        ) if clamp else (
            SMEx, SMNy
        )
    
    def scanmapXY_to_worldNE(self, X, Y):
        return (
            Y*self.resolution + self.ScanmapOriginOffset_Nym,
            X*self.resolution + self.ScanmapOriginOffset_Exm
        )

    def imprint(self, drone_curr_Ex_coord, drone_curr_Ny_coord, drone_yaw):

        rows, cols = self.scanmap.shape
        drone_scanmap_Ex_coord = drone_curr_Ex_coord - self.ScanmapOriginOffset_Exm
        drone_scanmap_Ny_coord = drone_curr_Ny_coord - self.ScanmapOriginOffset_Nym

        half_angle = np.arctan2(self.scanwidth / 2.0, self.scanradius) # Half-angle of the triangle at the tip

        # Total heading in radians
        heading_rad = np.deg2rad(90.0 - drone_yaw) #IMPT

        # World coordinates of every cell centre
        # Cell (r, c) has centre at world ((c + 0.5) * res, (r + 0.5) * res)
        # assuming row 0 = y=0, col 0 = x=0
        cell_r, cell_c = np.mgrid[0:rows, 0:cols]
        cell_x = (cell_c + 0.5) * self.resolution
        cell_y = (cell_r + 0.5) * self.resolution

        # Vector from tip to each cell centre
        dx = cell_x - drone_scanmap_Ex_coord
        dy = cell_y - drone_scanmap_Ny_coord

        # Distance and angle from tip to each cell
        dist       = np.hypot(dx, dy)
        cell_angle = np.arctan2(dy, dx)

        # Angular difference from heading, wrapped to [-π, π]
        angle_diff = (cell_angle - heading_rad + np.pi) % (2 * np.pi) - np.pi

        # Constraints
        in_range   = dist <= self.scanradius               # within triangle depth
        in_angle   = np.abs(angle_diff) <= half_angle      # within cone angle
        not_tip    = dist > 0                              # exclude the tip cell itself
        in_cone   = in_range & in_angle & not_tip & (self.scanmap == 0)

         # --- Ray-marching occlusion check ---
        cand_rows, cand_cols = np.where(in_cone)
        n_candidates = len(cand_rows)

        if n_candidates == 0:
            print("ray-marching failed, no candidates")
            # return False #dont return false first, rely on occupancy zones to trigger state.double_scanned_flag

        # Number of samples scales with ray length so no wall cell is skipped
        S = max(int(self.scanradius / self.resolution), 30)

        cand_x = (cand_cols + 0.5) * self.resolution
        cand_y = (cand_rows + 0.5) * self.resolution

        # Parametric samples t in (0.05, 0.95) — skip tip and cell itself
        t_vals = np.linspace(1.0/S, 1.0, S)[np.newaxis, :]               #NEW
        # t_vals = np.linspace(0.05, 0.95, S)[np.newaxis, :]      # (1, S) #OLD

        ray_x = (cand_x - drone_scanmap_Ex_coord)[:, np.newaxis] # (N, 1)
        ray_y = (cand_y - drone_scanmap_Ny_coord)[:, np.newaxis] # (N, 1)

        # World coords of every sample point — (N, S)
        sample_x = drone_scanmap_Ex_coord + t_vals * ray_x
        sample_y = drone_scanmap_Ny_coord + t_vals * ray_y

        # Convert sample world coords to grid indices — (N, S)
        sample_c = np.clip((sample_x / self.resolution).astype(int), 0, cols - 1)
        sample_r = np.clip((sample_y / self.resolution).astype(int), 0, rows - 1)

        # Check if any sample hits an obstacle (obstacle = 2 in your scanmap,
        # or swap in your self.grid / pointcloud occupancy array if separate)
        sample_vals = self.scanmap[sample_r, sample_c]
        occluded    = np.any(sample_vals == 2, axis=1)

        # Mark only unoccluded candidates
        visible_rows = cand_rows[~occluded]
        visible_cols = cand_cols[~occluded]\
        
        # self.scanmap[visible_rows, visible_cols] = 1 #ORIGINAL

        vals = self.scanmap[visible_rows, visible_cols]
        mask = (vals == 0) | (vals == 3)
        self.scanmap[visible_rows[mask], visible_cols[mask]] = 1 #ATTENTION MIGHT BE WRONG

        return True

    # # PUT POINTCLOUD OBSTACLES INTO SCANMAP - OLD
    # def blockoff_scanmap(self, obstaclecoords):
    #     x_idx = np.round(obstaclecoords[:, 0] / self.resolution).astype(int)
    #     y_idx = np.round(obstaclecoords[:, 1] / self.resolution).astype(int)
        
    #     # - self.ScanmapOriginOffset_Nym / self.resolution
    #     # - self.ScanmapOriginOffset_Exm / self.resolution

    #     #looks flipped but it worked previously
    #     row_clamped = np.clip(x_idx - self.ScanmapOriginOffset_Nym / self.resolution, 0, self.height-1)
    #     col_clamped = np.clip(y_idx - self.ScanmapOriginOffset_Exm / self.resolution, 0, self.width-1)
    # THIS OFFSET METHOD IS WRONG!!! SUBTRACT BEFORE ROUNDING EVERYTHING
    #     self.scanmap[row_clamped, col_clamped] = 2

    def blockoff_scanmap(self, obsNorth, obsEast):
        """
        Takes x2 1D numpy arrays obsNorth and obsEast that represent the N,E pos of obstacles in metres
        - blocks off the corresponding tiles in self.scanmap after offset and scaling
        """
        N_idx = np.round( (obsNorth - self.ScanmapOriginOffset_Nym) / self.resolution).astype(int)
        E_idx = np.round( (obsEast - self.ScanmapOriginOffset_Exm) / self.resolution).astype(int)

    #SWITCH
        # Clamp points so that they are still within the grid   //OPTION 1
        # row_clamped = np.clip(N_idx, 0, self.height-1)
        # col_clamped = np.clip(E_idx, 0, self.width-1)
        # self.scanmap[row_clamped, col_clamped] = 2

        # Keep only points that fall within the grid            //OPTION 2
        valid = (
            (N_idx >= 0) & (N_idx < self.height) &
            (E_idx >= 0) & (E_idx < self.width)
        )
    #SWITCH

        self.scanmap[N_idx[valid], E_idx[valid]] = 2

        #EDIT THIS FOR DILATION AND EROSION
        # # Extract binary mask where value == 5
        # mask = (gridmap == 5).astype(np.uint8) * 255

        # # Run morphology on the mask
        # cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        # # Write back — first clear all existing 5s, then stamp in the morphed result
        # gridmap[gridmap == 5] = 0
        # gridmap[cleaned_mask == 255] = 5


    #RETURNS GOAL CELL COORDINATE TUPLE (x,y)
    # def revaluate_frontiers(self, drone_Exm, drone_Nym, strategy=None, buffer=10):
    #     #blockoff_scanmp and imprint must be called before calling this
    #     #now that we know which cells are obstacles and free spaces, we can find the frontiers

    #     # PRE-PROCESSING # PRE-PROCESSING # PRE-PROCESSING
    #     frontiergrid = self.scanmap.copy()
    #     frontiergrid[frontiergrid>2] = 1 #reset all misc cells back to free cells
    #     y, x = np.ogrid[-buffer:buffer+1, -buffer:buffer+1]
    #     structure = x*x + y*y <= buffer*buffer
    #     frontiergrid[binary_dilation(frontiergrid == 2,structure=structure)] = 2 #inflate obstacles
    #     # PRE-PROCESSING # PRE-PROCESSING # PRE-PROCESSING

    #     frontier_mask = find_frontier_cells(frontiergrid) # Find frontier cells (raw boolean mask)
    #     clusters = find_frontier_clusters(frontier_mask, frontiergrid, min_cluster_size=10) # Find and rank clusters
    #     self.scanmap[frontier_mask] = 3 #frontier cell visual

    #     if not clusters:
    #         print("ALERT!!! no clusters found, returning just a list of frontier cell coords ")
    #         rowsNy, colsEx = np.where(self.scanmap == 3)
    #         return [self.scanmapXY_to_worldNE(int(colsEx[i]), int(rowsNy[i])) for i in range(len(rowsNy))]

    #     #ATTENTION WE ARE NOT SORTING FRONTIER CLUSTERS FOR NOW - RANKING IS DONE BASED ON THE CLOSEST CELL TO CENTROID
    #     # clusters = rank_frontier_clusters(
    #     #     clusters,
    #     #     (drone_Ny_u+self.drone_Ny_offset, drone_Ex_u+self.drone_Ex_offset),
    #     #     strategy)
        
    #     clusters = [closest_cell_to_centroid(cluster) for cluster in clusters]
    #     #so clusters is now a list of (Ny_u,Ex_u) goal coordinate tuples

    #     clusters = [self.scanmapXY_to_worldNE(goal_Ex_u, goal_Ny_u) for goal_Ny_u, goal_Ex_u in clusters]

    #     print(f"\n----------------\n{len(clusters)} CLUSTERS FOUND")
    #     return clusters # return tuple coords(Ny,Ex) in meters
    
    def searchlight(self, drone_curr_Exm_coord, drone_curr_Nym_coord):

        rows, cols = self.scanmap.shape
        drone_scanmap_Exm_coord = drone_curr_Exm_coord - self.ScanmapOriginOffset_Exm
        drone_scanmap_Nym_coord = drone_curr_Nym_coord - self.ScanmapOriginOffset_Nym

        # World coordinates of every cell centre
        cell_r, cell_c = np.mgrid[0:rows, 0:cols]
        cell_x = (cell_c + 0.5) * self.resolution
        cell_y = (cell_r + 0.5) * self.resolution

        # Distance from circle centre to every cell centre
        dist = np.hypot(
            cell_x - drone_scanmap_Exm_coord,
            cell_y - drone_scanmap_Nym_coord
        )

        #VISUALS: leave scanprint as long as cell is not an obstacle
        self.scanmap[(dist <= self.scanradius) & (self.scanmap != 2)] = 8