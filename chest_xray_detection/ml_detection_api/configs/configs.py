from pathlib import Path

from omegaconf import OmegaConf

INFERENCE_CONFIG = OmegaConf.load(Path(Path(__file__).parent, "inference_config.yaml"))
