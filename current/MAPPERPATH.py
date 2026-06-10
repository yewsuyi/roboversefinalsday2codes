# from ScanMap import ScanMapper
def expand_waypoints():
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
    
    waypoints = ( #WAYPOINTS XYU
        (58, 28), #START
        (17, 28), #bot left corner
        (17, 204), #top left corner
        (31, 204), #
        (31, 60), #
        (59, 60), #
        (59, 204), #
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
