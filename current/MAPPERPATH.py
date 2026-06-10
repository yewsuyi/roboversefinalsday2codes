# from ScanMap import ScanMapper
def expand_waypoints(careful=False):
    """
    Convert a list of waypoint coordinates into a complete path.

    Parameters
    ----------
    waypoints : list[tuple[int, int]]
        List of (x, y) waypoint coordinates.

    Returns
    -------
    list[tuple[int, int]]
        Complete path including all intermediate coordinates.
    """
    
    if careful:
        # +0.50M CLEARANCE
        waypoints = ( #WAYPOINTS XYU
            (58, 28), #START
            (27, 28), #bot left corner
            (27, 194), #top left corner
            (46, 194), #
            (46, 70), #
            (65, 70), #
            (65, 194), #
            (83, 194),
            (83, 28),
            (80, 28), #END
        )

    else:
        # VANILLA
        waypoints = ( #WAYPOINTS XYU
            (58, 28), #START
            (17, 28), #bot left corner
            (17, 204), #top left corner
            (42, 204), #
            (42, 60), #
            (67, 60), #
            (67, 204), #
            (93, 204),
            (93, 28),
            (80, 28), #END
        )

    if not waypoints:
        return []

    path = [waypoints[0]]

    for (x1, y1), (x2, y2) in zip(waypoints[:-1], waypoints[1:]):

        if x1 == x2:  # Vertical segment
            step = 1 if y2 > y1 else -1
            for y in range(y1 + step, y2 + step, step):
                path.append((x1, y))

        elif y1 == y2:  # Horizontal segment
            step = 1 if x2 > x1 else -1
            for x in range(x1 + step, x2 + step, step):
                path.append((x, y1))

        else:
            raise ValueError(
                f"Waypoints {(x1, y1)} and {(x2, y2)} are not aligned "
                "horizontally or vertically."
            )

    return path

if __name__ == "__main__":
    wp_xyu = expand_waypoints()
    # for x, y in wp_xyu:
    #     print(f"- {x},{y}")
