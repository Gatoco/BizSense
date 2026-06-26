from abc import ABC, abstractmethod
from ml.domain.models import Dataset


class DatasetRepository(ABC):
    @abstractmethod
    def save(self, dataset: Dataset) -> int:
        pass

    @abstractmethod
    def get_by_id(self, dataset_id: int, load_data: bool = False) -> Dataset:
        pass

    @abstractmethod
    def list_all(self) -> list[Dataset]:
        pass

    @abstractmethod
    def delete(self, dataset_id: int) -> None:
        pass