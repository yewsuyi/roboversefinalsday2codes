import time
from UWBaller import UWBParserThread, UWBxy_to_globalNE

TAG_ID = 0 #TODO

if __name__ == "__main__":

# TODO COPY THIS PART!!!!!! TODO #
    parser = UWBParserThread()
    if not parser.serial_port:
        print("No UWB device detected. Exiting.")
        # return
    else: parser.start()
# TODO COPY THIS PART!!!!!! TODO #

    try:
        while True:


            # TODO COPY THIS PART!!!!!! TODO #
            x, y, update_time, validity = parser.get_tag_position(TAG_ID) 

            if x is not None:

                N, E = UWBxy_to_globalNE(x, y)
                print(f"tag {TAG_ID}|UWBx:{x:.3f}|UWBy: {y:.3f}|N:{N:.3f}|E:{E:.3f}|VALID:{validity}|Last updated: {update_time}")
            
            # TODO COPY THIS PART!!!!!! TODO #

# IMPORTANT: WE WILL BE DEFINING OUR OWN NORTH:
# - According to the staff "north" (0 deg bearing) is wherever the drone faces when turned on
# - So we will align our "north" to one of the axes of the map (point drone in that dir every time)

# - If the UWB's x and y axes (let these be known as the J and K axes) do not align with the map axes,
# - we will have to rotate and somehow transform the axes

            time.sleep(1)

    except KeyboardInterrupt:
        parser.stop()
        parser.join()