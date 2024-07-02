from confz import BaseConfig, FileSource, validate_all_configs
from pydantic import BaseModel


class NetArchConfig(BaseModel):
    policy: list[int]
    value_function: list[int]


class IDNetConfig(BaseConfig):
    # Environment configs
    mim_data_path: str  # data for mimicking
    num_agents: int  # agents per unity process
    graphics: bool  # whether to show gui, i.e. headless mode
    executable_file: str | None  # path to unity executable
    use_subproc: bool  # use subprocesses for parallel environments
    n_proc: int  # number of subprocesses
    mt_shape: tuple[int, int]  # shape of the muscle torque

    # Training configs
    learning_rate: float
    seed: int

    # Network architecture configs
    net_arch: NetArchConfig

    # Training loop configs
    n_steps: int
    n_epochs: int
    total_timesteps: int

    tb_log_name: str

    CONFIG_SOURCES = FileSource(file="config/idnet_1024_512.yaml")


if __name__ == "__main__":
    validate_all_configs()
