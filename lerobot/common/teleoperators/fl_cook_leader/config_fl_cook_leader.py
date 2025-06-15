from dataclasses import dataclass

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("fl_cook_leader")
@dataclass
class FLCookLeaderConfig(TeleoperatorConfig):
    # Port to connect to the arms
    port_left_arm: str
    port_right_arm: str

    use_degrees: bool = False
