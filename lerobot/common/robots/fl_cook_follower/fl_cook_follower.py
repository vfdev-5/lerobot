import logging
import time
from functools import cached_property
from typing import Any

from lerobot.common.cameras.utils import make_cameras_from_configs
from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_fl_cook_follower import FLCookFollowerConfig

logger = logging.getLogger(__name__)


class FLCookFollower(Robot):
    """
    FabLab Cook Follower Arm designed by FabLab
    """

    config_class = FLCookFollowerConfig
    name = "fl_cook_follower"

    def __init__(self, config: FLCookFollowerConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100

        calib_left_arm = {
            key: value for key, value in self.calibration.items()
            if key.startswith("left_")
        }
        calib_right_arm = {
            key: value for key, value in self.calibration.items()
            if key.startswith("right_")
        }

        # so100 arm with gripper
        self.bus_left_arm = FeetechMotorsBus(
            port=self.config.port_left_arm,
            motors={
                "left_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "left_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "left_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "left_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "left_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calib_left_arm,
        )

        # so101 arm without gripper
        self.bus_right_arm = FeetechMotorsBus(
            port=self.config.port_right_arm,
            motors={
                "right_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "right_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "right_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "right_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "right_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                # "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calib_right_arm,
        )
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.bus_left_arm.motors
        } | {
            f"{motor}.pos": float for motor in self.bus_right_arm.motors
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return all([
            self.bus_left_arm.is_connected,
            self.bus_right_arm.is_connected,
            all(cam.is_connected for cam in self.cameras.values())
        ])

    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus_left_arm.connect()
        self.bus_right_arm.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus_left_arm.is_calibrated and self.bus_right_arm.is_calibrated

    def calibrate(self) -> None:

        self.calibration = {}
        for bus, descr in [
            (self.bus_left_arm, "Left Arm"),
            (self.bus_right_arm, "Right Arm")
        ]:
            logger.info(f"\nRunning calibration of {self} {descr}")
            bus.disable_torque()

            for motor in bus.motors:
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

            input(f"Move {self} {descr} to the middle of its range of motion and press ENTER....")
            homing_offsets = bus.set_half_turn_homings()

            print(
                "Move all joints sequentially through their entire ranges "
                "of motion.\nRecording positions. Press ENTER to stop..."
            )
            range_mins, range_maxes = bus.record_ranges_of_motion()

            calibration = {}
            for motor, m in bus.motors.items():
                calibration[motor] = MotorCalibration(
                    id=m.id,
                    drive_mode=0,
                    homing_offset=homing_offsets[motor],
                    range_min=range_mins[motor],
                    range_max=range_maxes[motor],
                )

            bus.write_calibration(calibration)
            self.calibration |= calibration

        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        for bus in [self.bus_left_arm, self.bus_right_arm]:
            with bus.torque_disabled():
                bus.configure_motors()
                for motor in bus.motors:
                    bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                    # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                    bus.write("P_Coefficient", motor, 16)
                    # Set I_Coefficient and D_Coefficient to default value 0 and 32
                    bus.write("I_Coefficient", motor, 0)
                    bus.write("D_Coefficient", motor, 32)

    def setup_motors(self) -> None:
        for bus, descr in [
            (self.bus_left_arm, "Left Arm"),
            (self.bus_right_arm, "Right Arm")
        ]:
            for motor in reversed(bus.motors):
                input(f"{descr}, Connect the controller board to the '{motor}' motor only and press enter.")
                bus.setup_motor(motor)
                print(f"'{motor}' motor id set to {bus.motors[motor].id}")

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        # Read arm position
        start = time.perf_counter()
        obs_dict1 = self.bus_left_arm.sync_read("Present_Position")
        obs_dict2 = self.bus_right_arm.sync_read("Present_Position")
        obs_dict = {
            f"{motor}.pos": val for motor, val in obs_dict1.items()
        } | {
            f"{motor}.pos": val for motor, val in obs_dict2.items()
        }
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            the action sent to the motors, potentially clipped.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        motors1 = {key + ".pos" for key in self.bus_left_arm.motors.keys()}
        motors2 = {key + ".pos" for key in self.bus_right_arm.motors.keys()}

        goal_pos1 = {
            key.removesuffix(".pos"): val for key, val in action.items()
            if key.endswith(".pos") and key in motors1
        }
        goal_pos2 = {
            key.removesuffix(".pos"): val for key, val in action.items()
            if key.endswith(".pos") and key in motors2
        }

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos1 = self.bus_left_arm.sync_read("Present_Position")
            present_pos2 = self.bus_right_arm.sync_read("Present_Position")
            goal_present_pos1 = {
                key: (g_pos, present_pos1[key]) for key, g_pos in goal_pos1.items()
            }
            goal_present_pos2 = {
                key: (g_pos, present_pos2[key]) for key, g_pos in goal_pos2.items()
            }
            goal_pos1 = ensure_safe_goal_position(goal_present_pos1, self.config.max_relative_target)
            goal_pos2 = ensure_safe_goal_position(goal_present_pos2, self.config.max_relative_target)

        # Send goal position to the arm
        self.bus_left_arm.sync_write("Goal_Position", goal_pos1)
        self.bus_right_arm.sync_write("Goal_Position", goal_pos2)
        return {
            f"{motor}.pos": val for motor, val in goal_pos1.items()
        } | {
            f"{motor}.pos": val for motor, val in goal_pos2.items()
        }

    def disconnect(self):
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        self.bus_left_arm.disconnect(self.config.disable_torque_on_disconnect)
        self.bus_right_arm.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
