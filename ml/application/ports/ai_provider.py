from abc import ABC, abstractmethod
from typing import List, Generator, Optional


class AIProvider(ABC):
    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        pass

    @abstractmethod
    def chat(self, messages: list, model: str, on_token=None) -> str:
        pass

    @abstractmethod
    def start_server(self) -> bool:
        pass

    @abstractmethod
    def pull_model(self, model: str) -> bool:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass