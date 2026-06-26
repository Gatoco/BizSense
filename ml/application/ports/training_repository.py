from abc import ABC, abstractmethod
from ml.domain.models import Training


class TrainingRepository(ABC):
    @abstractmethod
    def save(self, training: Training) -> int:
        pass

    @abstractmethod
    def get_by_id(self, training_id: int) -> Training:
        pass

    @abstractmethod
    def list_all(self) -> list[Training]:
        pass

    @abstractmethod
    def get_last(self) -> Training | None:
        pass