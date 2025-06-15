import logging
import time

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.common.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)

from ..teleoperator import Teleoperator
from .config_fl_cook_leader import FLCookLeaderConfig

logger = logging.getLogger(__name__)


class FLCookLeader(Teleoperator):
    """
    FabLab Cook Leader Arms designed by FabLab
    """

    config_class = FLCookLeaderConfig
    name = "fl_cook_leader"

    def __init__(self, config: FLCookLeaderConfig):
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

        self.bus_left_arm = FeetechMotorsBus(
            port=self.config.port_left_arm,
            motors={
                "left_shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "left_shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
                "left_elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
                "left_wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
                "left_wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
                "left_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=calib_left_arm,
        )

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

    @property
    def action_features(self) -> dict[str, type]:
        return {
            f"{motor}.pos": float for motor in self.bus_left_arm.motors
        } | {
            f"{motor}.pos": float for motor in self.bus_right_arm.motors
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus_left_arm.is_connected and self.bus_right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        self.bus_left_arm.connect()
        self.bus_right_arm.connect()
        if not self.is_calibrated and calibrate:
            self.calibrate()

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
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        for bus in [self.bus_left_arm, self.bus_right_arm]:
            bus.disable_torque()
            bus.configure_motors()
            for motor in bus.motors:
                bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    def setup_motors(self) -> None:
        for bus, descr in [
            (self.bus_left_arm, "Left Arm"),
            (self.bus_right_arm, "Right Arm")
        ]:
            for motor in reversed(bus.motors):
                input(f"{descr}, Connect the controller board to the '{motor}' motor only and press enter.")
                bus.setup_motor(motor)
                print(f"'{motor}' motor id set to {bus.motors[motor].id}")

    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action1 = self.bus_left_arm.sync_read("Present_Position")
        action1 = {f"{motor}.pos": val for motor, val in action1.items()}
        action2 = self.bus_right_arm.sync_read("Present_Position")
        action2 = {f"{motor}.pos": val for motor, val in action2.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        action = action1 | action2
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO(rcadene, aliberts): Implement force feedback
        raise NotImplementedError

    def disconnect(self) -> None:
        if not self.is_connected:
            DeviceNotConnectedError(f"{self} is not connected.")

        self.bus_left_arm.disconnect()
        self.bus_right_arm.disconnect()
        logger.info(f"{self} disconnected.")
