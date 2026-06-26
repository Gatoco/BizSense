from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


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
    extra: Optional[dict] = None


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
    id: Optional[int] = None
    name: str = ""
    filename: str = ""
    columns: List[str] = field(default_factory=list)
    rows: int = 0
    created_at: Optional[str] = None
    _data: Optional[dict] = None

    def get_column(self, col_name: str) -> List[float]:
        if self._data is None:
            raise ValueError("Dataset data not loaded")
        return self._data[col_name]

    def set_data(self, data: dict):
        self._data = data


@dataclass
class Training:
    id: Optional[int] = None
    dataset_id: int = 0
    dataset_name: str = ""
    model_type: str = ""
    x_col: str = ""
    y_col: str = ""
    alpha: float = 0.01
    iterations: int = 100
    theta_0: float = 0.0
    theta_1: float = 0.0
    final_cost: float = 0.0
    equation: str = ""
    created_at: Optional[str] = None
    history: Optional[List[IterationStep]] = None


@dataclass
class AppConfig:
    default_alpha: float = 0.01
    default_iterations: int = 100
    theme: str = "dark"
    language: str = "es"
    ai_provider: str = "auto"
    ai_model: str = "qwen2.5:1.5b"
    ai_endpoint: str = ""

    def to_dict(self) -> dict:
        return {
            'default_alpha': self.default_alpha,
            'default_iterations': self.default_iterations,
            'theme': self.theme,
            'language': self.language,
            'ai_provider': self.ai_provider,
            'ai_model': self.ai_model,
            'ai_endpoint': self.ai_endpoint
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AppConfig':
        return cls(
            default_alpha=float(data.get('default_alpha', 0.01)),
            default_iterations=int(data.get('default_iterations', 100)),
            theme=data.get('theme', 'dark'),
            language=data.get('language', 'es'),
            ai_provider=data.get('ai_provider', 'auto'),
            ai_model=data.get('ai_model', 'qwen2.5:1.5b'),
            ai_endpoint=data.get('ai_endpoint', '')
        )


@dataclass
class Stats:
    datasets_count: int = 0
    trainings_count: int = 0
    models_implemented: int = 4
    last_training: Optional[Training] = None


@dataclass
class InsightsMetrics:
    trend_direction: str = ""
    trend_rate: float = 0.0
    trend_rate_label: str = ""
    prediction_next: float = 0.0
    confidence: str = ""
    confidence_score: float = 0.0
    interpretation: str = ""
    model_type: str = ""
    dataset_name: str = ""
    dataset_rows: int = 0
    equation: str = ""
    x_col: str = ""
    y_col: str = ""
    x_data: List[float] = field(default_factory=list)
    y_data: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    future_predictions: List[float] = field(default_factory=list)
    future_x: List[float] = field(default_factory=list)


@dataclass
class ChatMessage:
    id: Optional[int] = None
    role: str = ""
    content: str = ""
    training_id: Optional[int] = None
    created_at: Optional[str] = None