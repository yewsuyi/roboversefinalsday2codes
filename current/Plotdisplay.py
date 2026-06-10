import asyncio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
matplotlib.use("TkAgg")

# from YwiGlobalMapper import GlobalMapper
# from customcodes.PositionGetters.get_position_with_task import SharedState
# from old.ScanMap import ScanMapper

async def pointcloud_plotter_task(
        mapper,
        state,
        scanmapper,
        stop_event: asyncio.Event,
        drawloopdelay:float = 0.5
    ):

    plt.ion()
#-----------------------------------------------------------
    # fig, ax = plt.subplots(figsize=(8, 8))
    
    # droneicon, = ax.plot([0], [0], "r*", markersize=12, label="Drone")
    # pts = mapper.get_global_points()
    # # dists = np.linalg.norm(pts, axis=1)
    # obstaclepcloud = ax.scatter(
    #     pts[:, 1],  # east
    #     pts[:, 0],  # north
    #     # c=dists,
    #     s=4,
    #     # cmap="viridis",
    #     c="grey",
    #     edgecolors="none",
    # )

    # ax.set_xlabel("East [m]")
    # ax.set_ylabel("North [m]")
    # ax.set_aspect("equal")
    # ax.grid(alpha=0.3)
    # ax.legend()
    # ax.ignore_existing_data_limits = True #for resizing later
    # plt.pause(0.05)
    # # plt.show()
    # fig.canvas.draw()

    # print("Pointcloud window up")
#-----------------------------------------------------------
    scanmap_cmap = ListedColormap([

        "grey",     # 0 - unknown
        "white",    # 1 - free
        "black",    # 2 - obstacle
        "purple",   # 3 - frontier cell
        "aqua",     # 4 - GOAL cell to pathfind to was red
        "blue",     # 5 - waypoints
        "green",    # 6 - OK aruco
        "yellow",   # 7 - curr drone pos
        "violet",   # 8 - hula searchlight
        "red",      # 9 - INVALID aruco

    ])
    CMAP_VMAX = 9 #max value that scanmap cells can be, UPDATE THIS IF ADDING NEW COLOURS

    fig2, ax2 = plt.subplots(figsize=(10, 10))

    scanmap_img = ax2.imshow(
        scanmapper.scanmap,
        cmap=scanmap_cmap,
        origin="lower",
        interpolation="nearest",
        vmin=0,
        vmax=CMAP_VMAX,
    )
    ax2.set_title("Scan Map")
    # print("Scanmap window up")
#-----------------------------------------------------------
    prev_val = 1
    print("Pointcloud & Scanmap plotter task started...")
    try:
        while True:
            if stop_event.is_set(): break
            # droneicon.set_data([float(state.east)], [float(state.north)])
            # pts = mapper.get_global_points()
            # if len(pts)>0: obstaclepcloud.set_offsets(np.column_stack([pts[:, 1], pts[:, 0]]))
            # # dists = np.linalg.norm(pts, axis=1)
            # # obstaclepcloud.set_array(dists) # updates the colour values
            # # obstaclepcloud.set_clim(vmin=dists.min(), vmax=dists.max()) # rescale colormap

            # ax.relim()
            # ax.update_datalim(obstaclepcloud.get_offsets())
            # ax.autoscale_view()
            # ax.margins(0.05)

            droneX, droneY = scanmapper.worldNE_to_scanmapXY(state.north, state.east)
            # prev_val = scanmapper.scanmap[droneY, droneX]
            # scanmapper.scanmap[droneY, droneX] = 7
            # scanmap_img.set_data(scanmapper.scanmap)
            # scanmapper.scanmap[droneY, droneX] = prev_val

            #square with radius 4
            row_min = max(0, droneY - 3)
            row_max = min(scanmapper.scanmap.shape[0], droneY + 1 + 3)
            col_min = max(0, droneX - 3)
            col_max = min(scanmapper.scanmap.shape[1], droneX + 1 + 3)

            prev_vals = scanmapper.scanmap[row_min:row_max, col_min:col_max].copy()
            scanmapper.scanmap[row_min:row_max, col_min:col_max] = 7
            scanmap_img.set_data(scanmapper.scanmap)
            scanmapper.scanmap[row_min:row_max, col_min:col_max] = prev_vals


            # fig.canvas.draw_idle()
            fig2.canvas.draw_idle()
            # fig.canvas.flush_events()
            fig2.canvas.flush_events()
            await asyncio.sleep(drawloopdelay)

    except asyncio.CancelledError: print("Pointcloud plotter-visualiser task cancelled.")
    except Exception as e: print(f"Pointcloud visualiser error: {type(e).__name__}: {e}")