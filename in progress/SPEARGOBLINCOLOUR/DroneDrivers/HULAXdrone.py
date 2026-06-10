from pyhulax import DroneAPI
from pyhulax.core import Direction, TelemetryUnavailable,CameraPitchMode
import asyncio
import math
from DroneDrivers.path_following import compute_path_follow_command, forward_speed_to_ned_velocity
from UWBaller import UWBxy_to_globalNE

dt = 0.05

class Drone():
    def __init__(
        self,
        USE_HULAX_MANUAL_MODE,
        UWB_TAG,
        USE_UWB_MODE,
        system_address="udpin://0.0.0.0:14540",
        takeoff_height=3.25,
        NE_TOLERANCE=0.05, #0.10?
        D_TOLERANCE=0.05,
        D_VELO_TOLERANCE=0.05,
        YAW_DEG_TOLERANCE=3.0,
        YAW_DEG_ANGVELO_TOLERANCE=2.0,
        MAX_VEL_NE=1.5,
        # TODO TUNE
        Kp = 0.1,
        Ki = 0,
        Kd = 0,
        # TODO TUNE

    ):
        self.uwb_tag = UWB_TAG
        self.uwb_mode = USE_UWB_MODE
        self.system_address = system_address
        self.takeoff_height = takeoff_height

        self.N_tolerance = NE_TOLERANCE
        self.E_tolerance = NE_TOLERANCE
        self.D_tolerance = D_TOLERANCE
        self.D_velo_tolerance = D_VELO_TOLERANCE
        # self.yaw_rad_tolerance = math.radians(YAW_DEG_TOLERANCE)
        self.yaw_deg_tolerance = YAW_DEG_TOLERANCE
        self.yaw_deg_angvelo_tolerance = YAW_DEG_ANGVELO_TOLERANCE

        self.NE_speedlimit = MAX_VEL_NE
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd



        self.drone = DroneAPI()
        self.hovertask = None
        self.ctrl = None
        if USE_HULAX_MANUAL_MODE:
            self.drone.set_app_mode(1)
            self.drone.set_velocity_level(round(self.NE_speedlimit*100)) #MUST BE BTWN 0 and 300, int

    def get_uwb_position_NE(self, UWBparser):
        x, y, update_time, validity = UWBparser.get_tag_position(self.uwb_tag)
        if x is None: return None, None, None
        
        N, E = UWBxy_to_globalNE(x, y)
        return N, E, validity

    async def get_position_ned(self):
        while True:
            try:
                east, north, up = self.drone.get_position()
                return north/100, east/100, -up/100
            
            except TelemetryUnavailable as e:
                print("position telemetry unavailable, trying again")
                await asyncio.sleep(0.05)

            except Exception as e: raise e #other error

    async def face_camera_down(self):
        self.drone.set_camera_angle(CameraPitchMode.DOWN_ABSOLUTE, 90)

    async def hovertask(self):
        try:
            while True:
                self.drone.send_manual_control()
                await asyncio.sleep(0.05)
        except asyncio.CancelledError: pass
    async def hover(self):
        if self.hovertask is None: self.hovertask = asyncio.create_task(self.hovertask())
        else: print("Already hovering.")
    async def stop_hover(self):
        if self.hovertask is not None:
            self.hovertask.cancel()  # Signal the worker loop to exit
            try:
                await self.hovertask  # 3. Give the event loop a tick to process the exit cleanly
            except asyncio.CancelledError:
                pass
            self.hovertask = None  # Safely wipe the handle after it is truly dead
            print("Hover task terminated.")



    def connect(self):
        self.drone.connect(self.system_address)
    def arm_and_takeoff(self, blocking):
        self.drone.takeoff(
            height_cm=round(self.takeoff_height * 100),
            blocking=blocking
        )
        
    def hover(self): self.drone.hover(duration_seconds=50.0, blocking=False) #prob no need use
    # i am assuming that any movement control will override the hover command, so no need for a stop_hover() fn
    def land(self): self.drone.land(blocking=False)

    def turn_left(self): self.drone.rotate(-90.0)
    def turn_right(self): self.drone.rotate(90.0)
    def u_turn(self): self.drone.rotate(180.0)

    # def flystraight(self, state, directionenum):
    # https://pyhulax.xenops.ae/reference/pyhulax/#pyhulax.DroneAPI.move, USE WITH BLOCKING=FALSE


    def prep_offboard(self):
        self.ctrl = self.drone.create_flight_controller()
        # ctrl.configure(
        #     kp_xy=2.5, # self.Kp
        #     kd_xy=0.5, # self.Kd
        #     position_tolerance_cm=3.0 # round(self.NE_tolerance*100)
        # )

    async def nonUWB_fly_to_position(self, UWBparser, target_N, target_E):
        # USE create_flight_controller
        self.stop_hover()
        if self.ctrl is None: self.prep_offboard()

        self.ctrl.set_target(target_E, target_N, self.takeoff_height)
        while not self.ctrl.has_converged():
            self.ctrl.update()
            await asyncio.sleep(0.05)
        self.ctrl.stop() #halt


        #TODO CAN REPEAT THIS WHOLE BLOCK FOR MORE ACCURACY
        #UWB correction
        if self.uwb_mode:
            while True:

                current_n, current_e, valid = self.get_uwb_position_NE(UWBparser)

                if valid is None:
                    print("UWB data MISSING, cannot navigate.")
                    await asyncio.sleep(dt)
                    break

                elif not valid:
                    print("UWB data OUTDATED, cannot navigate.")
                    await asyncio.sleep(dt)
                    break

            error_E = target_E - current_e
            error_N = target_N - current_n

            self.ctrl.set_target(target_E+error_E, target_N+error_N, self.takeoff_height)
            while not self.ctrl.has_converged():
                self.ctrl.update()
                await asyncio.sleep(0.05)
            self.ctrl.stop() #halt
        #TCAN REPEAT THIS WHOLE BLOCK FOR MORE ACCURACY
        
            
    # hula drone prob doesnt need to turn anyway
    # async def turn_to_yaw_deg(self):
    #     self.stop_hover()
    #     # BUT IF NEEDED, USE create_flight_controller, reference nonUWB_fly_to_position
    #     pass

    # NOTE USE_HULAX_MANUAL_MODE must be TRUE
    async def follow_waypoints(self):
        self.stop_hover()
        # USE send_manual_control at 20Hz
        # https://pyhulax.xenops.ae/sdk/pyhulax/#manual-control



    # NOTE USE_HULAX_MANUAL_MODE must be TRUE
    async def fly_to_position(self, UWBparser, target_Ny, target_Ex, state=None, yaw_deg_to_face=None, required_stable_samples=10):
        # USE send_manual_control at 20Hz
        # https://pyhulax.xenops.ae/sdk/pyhulax/#manual-control

        self.stop_hover()
        stable_counter = 0

        prev_err_n = None
        prev_err_e = None
        integral_n = 0.0
        integral_e = 0.0
        UWBskipped = False

        while True:

            if self.uwb_mode:

                current_n, current_e, valid = self.get_uwb_position_NE(UWBparser)

                if valid is None:
                    print("UWB data MISSING, cannot navigate.")
                    self.drone.send_manual_control()  # Stop movement if UWB data is not ready
                    UWBskipped = True
                    await asyncio.sleep(dt)
                    continue

                elif not valid:
                    print("UWB data OUTDATED, cannot navigate.")
                    self.drone.send_manual_control()  # Stop movement if UWB data is not ready
                    UWBskipped = True
                    await asyncio.sleep(dt)
                    continue

            else:

                # current_n = state.north #DIFF
                # current_e = state.east #DIFF
                current_n, current_e, current_d = await self.get_position_ned()

            #----------------------------------# UWB data is valid after this point

            err_n = target_Ny - current_n
            err_e = target_Ex - current_e
            if prev_err_n is None: prev_err_n = err_n # FOR CALCULATING INITIAL DERIVATIVE
            if prev_err_e is None: prev_err_e = err_e # FOR CALCULATING INITIAL DERIVATIVE

            if UWBskipped:
                prev_err_n = err_n
                prev_err_e = err_e
                UWBskipped = False

            # dist = math.sqrt(err_n**2 + err_e**2)
            vn = 0.0
            ve = 0.0

            # EXIT CONDITION
            if (
                abs(err_n) < self.N_tolerance and
                abs(err_e) < self.E_tolerance
            ):
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= required_stable_samples:
                self.drone.send_manual_control()
                print("Waypoint reached, hovering...")
                await self.hover()
                return
            # EXIT CONDITION
            
            integral_n += err_n * dt
            integral_e += err_e * dt
            derivative_n = (err_n - prev_err_n) / dt
            derivative_e = (err_e - prev_err_e) / dt
            prev_err_n = err_n
            prev_err_e = err_e

            if abs(err_n) < self.N_tolerance: vn = 0.0
            else: vn = self.Kp * err_n + self.Ki * integral_n + self.Kd * derivative_n

            if abs(err_e) < self.E_tolerance: ve = 0.0
            else: ve = self.Kp * err_e + self.Ki * integral_e + self.Kd * derivative_e

            # SPEED LIMIT
            horizontal_speed = math.sqrt(vn**2 + ve**2) #NOTE
            if horizontal_speed > self.NE_speedlimit:
                print(f"drone going faster than {self.NE_speedlimit}, limiting...")
                scale = self.NE_speedlimit / horizontal_speed
                vn *= scale
                ve *= scale
                # ── Anti-windup: undo the integral accumulation this tick ──
                integral_n -= err_n * dt
                integral_e -= err_e * dt
            # SPEED LIMIT
            
            # await self.send_velocity(vn, ve, 0.0, yaw_deg_to_face)
            # vn and ve are speeds in m/s, i have done the math
            self.drone.send_manual_control(vn/self.NE_speedlimit, ve/self.NE_speedlimit)
            # NOTE THAT IF DRONE TURNS, NEED TO ROTATE THIS SINCE
            # - send_manual_control is (fwd, right) and not (N, E)
            # - but it works in this case if the drone is just oriented north forever

            await asyncio.sleep(dt) #20Hz


