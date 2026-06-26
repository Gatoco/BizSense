from dataclasses import dataclass
from typing import List


@dataclass
class IterationStep:
    iteration: int
    theta_0: float
    theta_1: float
    theta_0_real: float
    theta_1_real: float
    cost: float
    gradient_0: float
    gradient_1: float
    predictions: List[float]


@dataclass
class TrainingResult:
    theta: List[float]
    history: List[IterationStep]
    x_data: List[float]
    y_data: List[float]
    x_mean: float
    x_std: float


@dataclass
class Dataset:
    columns: List[str]
    data: dict
    rows: int
    
    def get_column(self, col_name: str) -> List[float]:
        return self.data[col_name]