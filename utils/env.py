from dataclasses import dataclass


@dataclass
class EnvConfig:
    input_dir: str = "/home/ryo52/work-kaggle/csiro/input"
    output_dir: str = "/home/ryo52/work-kaggle/csiro/output"
    artifacts_dir: str = "/home/ryo52/work-kaggle/csiro/output/artifacts"