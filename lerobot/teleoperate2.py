# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple script to control two robots from teleoperation.

Example:

```shell
python -m lerobot.teleoperate_2 \
    --robot1.type=so101_follower \
    --robot1.port=/dev/ttyACM0 \
    --robot1.id=black1 \
    --robot2.type=so101_follower \
    --robot2.port=/dev/ttyACM1 \
    --robot2.id=black2 \
    --teleop1.type=so101_leader \
    --teleop1.port=/dev/ttyACM2 \
    --teleop1.id=blue1
    --teleop2.type=so101_leader \
    --teleop2.port=/dev/ttyACM3 \
    --teleop2.id=blue2
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pprint import pformat

import draccus
import numpy as np
import rerun as rr

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.common.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    koch_follower,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.common.teleoperators import (
    Teleoperator,
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.common.utils.robot_utils import busy_wait
from lerobot.common.utils.utils import init_logging, move_cursor_up
from lerobot.common.utils.visualization_utils import _init_rerun

from .common.teleoperators import gamepad, koch_leader, so100_leader, so101_leader  # noqa: F401


@dataclass
class Teleoperate2Config:
    teleop1: TeleoperatorConfig
    teleop2: TeleoperatorConfig
    robot1: RobotConfig
    robot2: RobotConfig
    # Limit the maximum frames per second.
    fps: int = 60
    teleop_time_s: float | None = None
    # Display all cameras on screen
    display_data: bool = False


def teleop_loop(
    teleop1: Teleoperator,
    robot1: Robot,
    teleop2: Teleoperator,
    robot2: Robot,
    fps: int,
    display_data: bool = False,
    duration: float | None = None
):
    display_len = max([
        max(len(key) for key in robot1.action_features),
        max(len(key) for key in robot2.action_features),
    ])
    start = time.perf_counter()
    while True:
        loop_start = time.perf_counter()

        action1 = teleop1.get_action()
        action2 = teleop2.get_action()

        if display_data:
            observation1 = robot1.get_observation()
            observation2 = robot2.get_observation()

            for obs, val in observation1.items():
                if isinstance(val, float):
                    rr.log(f"1 observation_{obs}", rr.Scalars(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"1 observation_{obs}", rr.Image(val), static=True)
            for act, val in action1.items():
                if isinstance(val, float):
                    rr.log(f"1 action_{act}", rr.Scalars(val))
            for obs, val in observation2.items():
                if isinstance(val, float):
                    rr.log(f"2 observation_{obs}", rr.Scalars(val))
                elif isinstance(val, np.ndarray):
                    rr.log(f"2 observation_{obs}", rr.Image(val), static=True)
            for act, val in action1.items():
                if isinstance(val, float):
                    rr.log(f"2 action_{act}", rr.Scalars(val))

        robot1.send_action(action1)
        robot2.send_action(action2)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)

        loop_s = time.perf_counter() - loop_start

        print("\n" + "-" * (display_len + 10))
        print(f"{'NAME':<{display_len}} | {'NORM':>7}")
        for motor, value in action1.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        for motor, value in action2.items():
            print(f"{motor:<{display_len}} | {value:>7.2f}")
        print(f"\ntime: {loop_s * 1e3:.2f}ms ({1 / loop_s:.0f} Hz)")

        if duration is not None and time.perf_counter() - start >= duration:
            return

        move_cursor_up(len(action1) + 5)
        move_cursor_up(len(action2) + 5)


@draccus.wrap()
def teleoperate2(cfg: Teleoperate2Config):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        _init_rerun(session_name="teleoperation")

    teleop1 = make_teleoperator_from_config(cfg.teleop1)
    teleop2 = make_teleoperator_from_config(cfg.teleop2)
    robot1 = make_robot_from_config(cfg.robot1)
    robot2 = make_robot_from_config(cfg.robot2)

    teleop1.connect()
    teleop2.connect()
    robot1.connect()
    robot2.connect()

    try:
        teleop_loop(
            teleop1,
            robot1,
            teleop2,
            robot2,
            cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s
        )
    except KeyboardInterrupt:
        pass
    finally:
        if cfg.display_data:
            rr.rerun_shutdown()
        teleop1.disconnect()
        robot1.disconnect()
        teleop2.disconnect()
        robot2.disconnect()


if __name__ == "__main__":
    teleoperate2()
