from dataclasses import dataclass


@dataclass
class Config:
    enable_validation: bool = True


config = Config()
