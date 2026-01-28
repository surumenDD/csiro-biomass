from dataclasses import dataclass


@dataclass
class EnvConfig:
    experiments_dir = "/home/ryo52/work-kaggle/csiro/experiments"
    input_dir: str = "/home/ryo52/work-kaggle/csiro/input"
    output_dir: str = "/home/ryo52/work-kaggle/csiro/output"
    artifacts_dir: str = "/home/ryo52/work-kaggle/csiro/artifacts/experiments"