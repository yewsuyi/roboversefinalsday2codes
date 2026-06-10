# uses the code given by the organisers, gemini says theres alot of critical issues with serial reading
import threading
import serial
import struct
import serial.tools.list_ports
import time

class UWBParserThread(threading.Thread):
    def __init__(
        self,
        x_origin=0.0,
        y_origin=0.0,
        serial_port=None,   # NOTE
        baud_rate=921600,   # NOTE
        max_acceptable_timediff_s=0.5, #TODO TUNE
    ):
        super().__init__()
        self.serial_port = serial_port if serial_port else self.detect_com_port()
        self.baud_rate = baud_rate
        self.data_lock = threading.Lock()
        self.tag_data = {}  # Stores tag data {tag_id: (x, y, update_time)}
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.MATDs = max_acceptable_timediff_s
        self.running = True

    def detect_com_port(self):
        """Detects the COM port of the connected USB device."""
        ports = serial.tools.list_ports.comports()
        for port in ports:
            if "USB" in port.description:  # Check for USB device
                print(f"Detected UWB Device on {port.device}")
                return port.device
        print("No UWB device detected.")
        return None

    def run(self):
        """Thread execution function."""
        if not self.serial_port:
            print("No valid COM port detected. Exiting thread.")
            return
        
# EDITED PORTION # EDITED PORTION # EDITED PORTION # EDITED PORTION # EDITED PORTION #
        try:
            with serial.Serial(self.serial_port, self.baud_rate, timeout=0.1) as ser:
                buffer = bytearray()
                ser.reset_input_buffer() # Clear old buffers on startup
                
                while self.running:
                    # Read whatever is ready in the serial buffer cache
                    avail = ser.in_waiting
                    chunks = ser.read(avail if avail > 0 else 1)
                    if not chunks:
                        continue
                    
                    buffer.extend(chunks)
                    
                    # Robust Sliding Window parsing engine
                    while len(buffer) >= 896:
                        # 0x55 is hex for ord('U')
                        if buffer[0] == 0x55 and buffer[1] == 0x00:
                            if buffer[895] == 0xEE:
                                self.parse_data(buffer[:896])
                                del buffer[:896] # Cleanly consume packet
                            else:
                                # Misaligned packet window footprint, advance frame safely
                                del buffer[0]
                        else:
                            # Drop leading byte and keep shifting until headers match
                            del buffer[0]
# EDITED PORTION # EDITED PORTION # EDITED PORTION # EDITED PORTION # EDITED PORTION #

        except serial.SerialException as e:
            print(f"Serial Error: {e}")
            self.running = False

    def parse_data(self, data):
        """Parses the received serial data and updates the tag data."""
        if len(data) < 896:
            print("Insufficient data to parse.")
            return

        frame_header, function_mark = struct.unpack("<BB", data[:2])
        if frame_header != 0x55 or function_mark != 0x00:
            print("Invalid Frame Header or Function Mark")
            return
        
        offset = 2
        # self.tag_data = {} REMOVED (whats the point???) # NOTE NOTE NOTE TODO CHECK
        with self.data_lock:
            for _ in range(30):
                if data[offset] != 0xFF:
                    block_id, role = struct.unpack("<BB", data[offset:offset + 2])
                    offset += 2
                    pos_x = int.from_bytes(data[offset:offset + 3], 'little', signed=True) / 1000
                    pos_y = int.from_bytes(data[offset + 3:offset + 6], 'little', signed=True) / 1000
                    offset += 9  # Skip Z position
                    offset += 16  # Skip distance values

                    self.tag_data[block_id] = (pos_x, pos_y, time.time())


                else:
                    offset += 27

    def get_tag_position(self, tag_id):
        """Returns the x, y position, update time, and VALIDNESS (NEW) of a specific tag."""
        with self.data_lock:

            posx, posy, updatetime = self.tag_data.get(tag_id, (None, None, None))

            if posx is None: return (None, None, None, None)

            timediff = time.time() - updatetime
            validity = timediff < self.MATDs
            if not validity: print(f"UWBallerSerial: UWB INVALID, delayed by {timediff:.2f}s")

            return posx, posy, updatetime, validity

    def stop(self):
        """Stops the thread."""
        self.running = False