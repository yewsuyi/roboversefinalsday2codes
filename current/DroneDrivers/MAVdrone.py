import asyncio
import math
from mavsdk import System
from mavsdk.offboard import Offboard, OffboardError
from mavsdk.offboard import VelocityNedYaw, PositionNedYaw
from mavsdk.action import ActionError
from DroneDrivers.path_following import compute_path_follow_command, forward_speed_to_ned_velocity, slew_yaw
from UWBaller import UWBxy_to_globalNE

dt = 0.05 #since setpoint spam is at 20Hz (fly_to_position)

class Drone:
    def __init__(
        self,
        UWB_TAG,
        USE_UWB_MODE,
        system_address="udpin://0.0.0.0:14540",
        takeoff_height=3.25,

        # TODO TUNE
        NE_TOLERANCE=0.05, #0.10?
        D_TOLERANCE=0.05,
        D_VELO_TOLERANCE=0.15, #0.05
        YAW_DEG_TOLERANCE=3.0,
        YAW_DEG_ANGVELO_TOLERANCE=3.0, #2.0,
        MAX_VEL_NE=1.5,
        
        Kp = 0.7, #0.1
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


        self.drone = System()
        self.hovertask = None

    def get_uwb_position_NE(self, UWBparser):
        x, y, update_time, validity = UWBparser.get_tag_position(self.uwb_tag)
        if x is None: return None, None, None
        
        N, E = UWBxy_to_globalNE(x, y)
        return N, E, validity
        

    
    async def _hovertask(self, state):
        try:
            while True:
                await self.send_velocity(0.0, 0.0, 0.0, state.yaw_deg)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError: pass
    async def hover(self, state):
        if self.hovertask is None: self.hovertask = asyncio.create_task(self._hovertask(state))
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
    

    async def flystraight(self, state, directionenum):
        if directionenum == 0: await self.send_velocity(state.explorespeed,0.0, 0.0, state.yaw_deg)
        elif directionenum == 1: await self.send_velocity(0.0, state.explorespeed, 0.0, state.yaw_deg)
        elif directionenum == 2: await self.send_velocity(-state.explorespeed,0.0, 0.0, state.yaw_deg)
        elif directionenum == 3: await self.send_velocity(0.0, -state.explorespeed, 0.0, state.yaw_deg)

    async def turn_left(self, state): pass
    async def turn_right(self, state): pass
    async def u_turn(self, state): pass
# ================================================================================= #

    async def fly_to_position(self, UWBparser, target_Ny, target_Ex, state, yaw_deg_to_face, required_stable_samples=10):

        await self.stop_hover()
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
                    await self.send_velocity(0.0, 0.0, 0.0, yaw_deg_to_face)  # Stop movement if UWB data is not ready
                    UWBskipped = True
                    await asyncio.sleep(dt)
                    continue

                elif not valid:
                    print("UWB data OUTDATED, cannot navigate.")
                    await self.send_velocity(0.0, 0.0, 0.0, yaw_deg_to_face)  # Stop movement if UWB data is not ready
                    UWBskipped = True
                    await asyncio.sleep(dt)
                    continue

            else:

                current_n = state.north
                current_e = state.east

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
                await self.send_velocity(0.0, 0.0, 0.0, yaw_deg_to_face)
                print("Waypoint reached, hovering...")
                await self.hover(state)
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
            horizontal_speed = math.sqrt(vn**2 + ve**2)
            if horizontal_speed > self.NE_speedlimit:
                print(f"drone going faster than {self.NE_speedlimit}, limiting...")
                scale = self.NE_speedlimit / horizontal_speed
                vn *= scale
                ve *= scale
                # ── Anti-windup: undo the integral accumulation this tick ──
                integral_n -= err_n * dt
                integral_e -= err_e * dt
            # SPEED LIMIT
            
            await self.send_velocity(vn, ve, 0.0, yaw_deg_to_face)

            await asyncio.sleep(dt) #20Hz
            # await asyncio.sleep(0.1) #10Hz (given)


    #send setpoints until it brakes, uses the same velocity tolerance as fly_to_position
    async def velocity_brake(self, state):
        #spam set_velocity_ned(0.0) until both north velo and east velo are below a threshold, then return
        
        for i in range(10):
            await self.send_velocity(0.0, 0.0, 0.0, state.yaw_deg)
            await asyncio.sleep(0.05)

        print("Assumed stabilised")

    #NO NEED PID, JUST SEND SETPOINT
    async def turn_to_yaw_deg(
        self,
        state,
        target_yaw_deg,
        required_stable_samples=10
    ):
        """
        Makes drone rotate using setpoint spam, returns when done rotating
        """
        await self.stop_hover()
        stable_counter = 0

        while True:

            current_yaw = state.yaw_deg
            yaw_rate = state.yaw_deg_angvelo

            # Smallest signed angle difference
            yaw_error = (
                (target_yaw_deg - current_yaw + 180)
                % 360
                - 180
            )

            await self.send_position_setpoint(state.north, state.east, state.down, target_yaw_deg)

            if (
                abs(yaw_error) < self.yaw_deg_tolerance
                and abs(yaw_rate) < self.yaw_deg_angvelo_tolerance
            ):
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= required_stable_samples:
                print(f"Yaw stabilized to {target_yaw_deg}")
                return

            await asyncio.sleep(0.05) #20 Hz update

# ================================================================================= #

#LINE FOLLOWER

    async def follow_waypoints(
        self,
        scanmapper,
        UWBparser,
        waypoints,
        state,
        goal_xy_coords,

        navspeed=0.3,###1.5, ###1.0, #0.5
        goal_threshold=0.10, #0.30 ##0.15, #1.3, #1.5 ####1.0 ##0.75 #0.30 #0.30
        waypoint_tolerance=0.10, ##0.10, #0.60,###0.50 ##0.40 #0.20 #0.05
        use_yaw_slowdown=True,
        slow_yaw_error_deg=20.0, #30.0, ####15.0, ###15.0, ##20.0
        stop_yaw_error_deg=40.0, #60.0, ####30.0, ###30.0, ##40.0

        loopdelay=0.05,
        max_yaw_rate_deg_per_tick = 2.5, #2.0, #4.0, #3.0 #2.5 ###2.0 ##4.0 #10.0  # tune this — degrees per loopdelay tick
        lookahead=0.60, ##0.40, #2.0,####1.5, ###1.0
        
    ):
        """
        Follow (x, y) waypoints using MAVSDK offboard velocity commands.
        - x = east, y = north
        - yaw 0 = north, yaw 90 = east
        - caller must start offboard mode before calling this
        """
        commanded_yaw = state.yaw_deg  # initialise to current yaw
        mypath = waypoints.copy()
        

        try:

            await self.stop_hover()
            while True:

                if self.uwb_mode:

                    current_n, current_e, valid = self.get_uwb_position_NE(UWBparser)

                    if valid is None:
                        print("UWB data MISSING, cannot follow path.")
                        await self.send_velocity(0.0, 0.0, 0.0, state.yaw_deg)  # Stop movement if UWB data is not ready
                        # UWBskipped = True
                        await asyncio.sleep(loopdelay)
                        continue

                    elif not valid:
                        print("UWB data OUTDATED, cannot follow path.")
                        await self.send_velocity(0.0, 0.0, 0.0, state.yaw_deg)  # Stop movement if UWB data is not ready
                        # UWBskipped = True
                        await asyncio.sleep(loopdelay)
                        continue

                    drone_position = (current_e, current_n)

                else: drone_position = (state.east, state.north)

                drone_yaw_deg = state.yaw_deg

                speed_multiplier, target_yaw, distance_to_goal, target_position_xym = compute_path_follow_command(
                    waypoints=mypath,
                    drone_position=drone_position,
                    drone_yaw_deg=drone_yaw_deg,
                    goal_position=goal_xy_coords,
                    waypoint_tolerance=waypoint_tolerance,
                    goal_tolerance=goal_threshold,
                    use_yaw_slowdown=use_yaw_slowdown,
                    slow_yaw_error_deg=slow_yaw_error_deg,
                    stop_yaw_error_deg=stop_yaw_error_deg,
                    lookahead=lookahead,
                )

                target_xm, target_ym = target_position_xym
                target_xu, target_yu = scanmapper.worldNE_to_scanmapXY(target_ym, target_xm)
                scanmapper.scanmap[target_yu, target_xu] = 4

                if distance_to_goal <= goal_threshold:
                    # await self.velocity_brake(state)
                    await self.hover(state)
                    # self.send_velocity(0.0, 0.0, 0.0, commanded_yaw)
                    return

                #LERP the YAW
                commanded_yaw = slew_yaw(commanded_yaw, target_yaw, max_yaw_rate_deg_per_tick)

                forward_speed = navspeed * speed_multiplier
                north_velocity, east_velocity = forward_speed_to_ned_velocity(
                    forward_speed=forward_speed,
                    yaw_deg=commanded_yaw,
                )

                await self.send_velocity(
                    north_velocity,
                    east_velocity,
                    0.0,
                    commanded_yaw,
                )

                await asyncio.sleep(loopdelay)

        except Exception as e:
            print(f"follow_waypoints error: {type(e).__name__}: {e}")
            raise



# ================================================================================= #

    async def send_velocity(self, vNy, vEx, vDOWN, yaw_deg):
        #  yaw_deg = yaw_deg + 5.15662
            await self.drone.offboard.set_velocity_ned(
                VelocityNedYaw(north_m_s=vNy, east_m_s=vEx, down_m_s=vDOWN, yaw_deg=yaw_deg)
        )

    async def send_position_setpoint(self, posNy, posEx, posDOWN, yaw_deg):
        await self.drone.offboard.set_position_ned(
                PositionNedYaw(north_m=posNy, east_m=posEx, down_m=posDOWN, yaw_deg=yaw_deg)
        )


    async def prep_offboard(self, yaw_deg):
        for attempt in range(1,11):
            try:
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, float(yaw_deg)))
                await asyncio.sleep(0.05)
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, float(yaw_deg)))
                await asyncio.sleep(0.05)
                await self.drone.offboard.start()
                # await asyncio.sleep(0.01)
                return
            except OffboardError as e:
                print(f"Offboard start failed (attempt {attempt}): {e}")
                await self.drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, float(yaw_deg)))
                await asyncio.sleep(0.1)

    async def NEW_prep_offboard(self):
        # PRE-STREAM INITIAL SETPOINT (Mandatory for Offboard)
        # Zero movement constraint to safely bridge into offboard mode
        initial_pos = PositionNedYaw(0.0, 0.0, 0.0, 0.0)
        initial_vel = VelocityNedYaw(0.0, 0.0, 0.0, 0.0)
        await self.drone.offboard.set_position_velocity_ned(initial_pos, initial_vel)

        print("-- Engaging Offboard Mode")
        try:
            await self.drone.offboard.start()
        except OffboardError as error:
            print(f"Offboard failed: {error._result.result}")
            await self.drone.action.disarm()
            return

    async def stop_offboard(self):
        await self.drone.offboard.stop()
        await asyncio.sleep(0.05)

# ================================================================================= #

    async def wait_for_takeoff_stable(
        self,
        state,
        additional_height,
        stable_time=0.5
    ):
        """
        Wait until drone reaches target height and stops climbing.
        Uses PX4 NED coordinates:
            Down is positive
            Up is negative
        """

        target_d = -self.takeoff_height

        stable_counter = 0
        required_samples = int(stable_time / 0.05)

        while True:

            current_d = state.down
            current_vd = state.down_velo

            altitude_error = abs(current_d - target_d)

            # await self.send_position_setpoint(state.north, state.east, target_d, 0.0)
            await self.send_position_setpoint(state.north, state.east, target_d-additional_height, 90.0)

            if (
                altitude_error < self.D_tolerance
                and abs(current_vd) < self.D_velo_tolerance
            ):
                stable_counter += 1
            else:
                stable_counter = 0

            if stable_counter >= required_samples:
                print("Takeoff altitude stabilized")
                return

            await asyncio.sleep(0.05)

# ================================================================================= #

    async def arm_and_takeoff(self):

        await self.NEW_wait_until_ready()

        # try: await self.drone.action.arm()
        # except ActionError as e: raise RuntimeError(f"Arm failed: {e}") from e

    # OLD WORKING CODE
        # try: await self.drone.action.set_takeoff_altitude(takeoff_height)
        # except Exception: pass

        # try: await self.drone.action.takeoff()
        # except ActionError as e: raise RuntimeError(f"Takeoff failed: {e}") from e

        # print("Taking off...")
        # await asyncio.sleep(8)


        try:
            await self.drone.action.set_takeoff_altitude(self.takeoff_height)
            await self.drone.action.arm()
            await self.drone.action.takeoff()

            # Wait until airborne and stable
            async for in_air in self.drone.telemetry.in_air():
                if in_air:
                    break
                await asyncio.sleep(0.05)

            await asyncio.sleep(2.0)

        except Exception as e: raise RuntimeError(f"NEW takeoff code CRASHED: {e}") from e

    async def connect(self):
        await self.drone.connect(system_address=self.system_address)

        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("Connected")
                break

    async def NEW_wait_until_ready(self, timeout_s=30.0):
        # OPTICAL FLOW HEALTH CHECK
        print("Waiting for Optical Flow / Local Position Lock...")
        async for health in self.drone.telemetry.health():
            if health.is_local_position_ok:
                print("✔ Optical Flow initialized! Local position estimate is healthy.")
                break

    async def wait_until_ready(self, timeout_s=30.0):
        """
        Be less strict than requiring global+home.
        Accept armable, or local position OK, or global+home.
        """
        loop = asyncio.get_running_loop()
        start = loop.time()

        async for health in self.drone.telemetry.health():
            armable = getattr(health, "is_armable", False)
            local_ok = getattr(health, "is_local_position_ok", False)
            global_ok = getattr(health, "is_global_position_ok", False)
            home_ok = getattr(health, "is_home_position_ok", False)

            print(
                f"Health: armable={armable}, "
                f"local_ok={local_ok}, global_ok={global_ok}, home_ok={home_ok}"
            )

            if armable or local_ok or (global_ok and home_ok):
                print("Vehicle readiness condition satisfied")
                return

            if loop.time() - start > timeout_s:
                raise TimeoutError(
                    "Timed out waiting for readiness "
                    f"(armable={armable}, local_ok={local_ok}, "
                    f"global_ok={global_ok}, home_ok={home_ok})"
                )
            
    async def land(self):
        try:
            print("Land")
            await self.drone.action.land()
            # await asyncio.sleep(8)
        except Exception as e: print(f"Landing failed: {e}")