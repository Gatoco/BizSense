from datetime import datetime
import pandas as pd
import numpy as np
from ml.domain.models import Dataset, Training, TrainingResult, AppConfig, Stats
from ml.domain.services import gradient_descent
from ml.application.ports.dataset_repository import DatasetRepository
from ml.application.ports.training_repository import TrainingRepository
from ml.application.ports.config_repository import ConfigRepository


class UploadDataUseCase:
    def __init__(self, dataset_repo: DatasetRepository):
        self.dataset_repo = dataset_repo

    def execute(self, filename: str, file_content) -> Dataset:
        df = pd.read_csv(file_content)
        data = {col: df[col].tolist() for col in df.columns}

        dataset = Dataset(
            name=filename,
            filename=filename,
            columns=list(df.columns),
            rows=len(df),
            created_at=datetime.now().isoformat(timespec='seconds')
        )
        dataset.set_data(data)

        dataset.id = self.dataset_repo.save(dataset)
        return dataset


class ListDatasetsUseCase:
    def __init__(self, dataset_repo: DatasetRepository):
        self.dataset_repo = dataset_repo

    def execute(self) -> list[Dataset]:
        return self.dataset_repo.list_all()


class GetDatasetUseCase:
    def __init__(self, dataset_repo: DatasetRepository):
        self.dataset_repo = dataset_repo

    def execute(self, dataset_id: int, load_data: bool = False) -> Dataset:
        return self.dataset_repo.get_by_id(dataset_id, load_data=load_data)


class DeleteDatasetUseCase:
    def __init__(self, dataset_repo: DatasetRepository):
        self.dataset_repo = dataset_repo

    def execute(self, dataset_id: int) -> None:
        self.dataset_repo.delete(dataset_id)


class TrainModelUseCase:
    def __init__(self, dataset_repo: DatasetRepository, training_repo: TrainingRepository):
        self.dataset_repo = dataset_repo
        self.training_repo = training_repo

    def execute(self, dataset_id: int, x_col: str, y_col: str, alpha: float, iterations: int) -> TrainingResult:
        dataset = self.dataset_repo.get_by_id(dataset_id, load_data=True)

        x_data = dataset.get_column(x_col)
        y_data = dataset.get_column(y_col)

        theta, history, x_mean, x_std = gradient_descent(x_data, y_data, alpha, iterations)

        training = Training(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            model_type='linear_regression',
            x_col=x_col,
            y_col=y_col,
            alpha=alpha,
            iterations=iterations,
            theta_0=theta[0],
            theta_1=theta[1],
            final_cost=history[-1].cost if history else 0,
            equation=f"y = {theta[0]:.4f} + {theta[1]:.4f} * x",
            created_at=datetime.now().isoformat(timespec='seconds')
        )
        training.history = history
        training.id = self.training_repo.save(training)

        return TrainingResult(
            theta=theta,
            history=history,
            x_data=x_data,
            y_data=y_data,
            x_mean=x_mean,
            x_std=x_std
        )


class GetTrainingHistoryUseCase:
    def __init__(self, training_repo: TrainingRepository):
        self.training_repo = training_repo

    def execute(self) -> list[Training]:
        return self.training_repo.list_all()


class GetStatsUseCase:
    def __init__(self, dataset_repo: DatasetRepository, training_repo: TrainingRepository):
        self.dataset_repo = dataset_repo
        self.training_repo = training_repo

    def execute(self) -> Stats:
        datasets = self.dataset_repo.list_all()
        trainings = self.training_repo.list_all()
        last = self.training_repo.get_last()

        return Stats(
            datasets_count=len(datasets),
            trainings_count=len(trainings),
            models_implemented=1,
            last_training=last
        )


class ManageConfigUseCase:
    def __init__(self, config_repo: ConfigRepository):
        self.config_repo = config_repo

    def get(self) -> AppConfig:
        return self.config_repo.get_all()

    def update(self, config: AppConfig) -> None:
        self.config_repo.update(config)


class PredictUseCase:
    def execute(self, x_values: list, theta: list) -> list:
        x = np.array(x_values, dtype=float)
        t = np.array(theta, dtype=float)
        predictions = t[0] + t[1] * x
        return predictions.tolist()