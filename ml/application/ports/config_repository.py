from abc import ABC, abstractmethod
from ml.domain.models import AppConfig


class ConfigRepository(ABC):
    @abstractmethod
    def get_all(self) -> AppConfig:
        pass

    @abstractmethod
    def update(self, config: AppConfig) -> None:
        pass