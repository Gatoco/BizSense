from ml.domain.models import Dataset, TrainingResult
from ml.domain.services import gradient_descent


class TrainModelUseCase:
    def execute(self, dataset: Dataset, x_col: str, y_col: str, alpha: float = 0.01, iterations: int = 100) -> TrainingResult:
        x_data = dataset.get_column(x_col)
        y_data = dataset.get_column(y_col)
        
        theta, history, x_mean, x_std = gradient_descent(x_data, y_data, alpha, iterations)
        
        return TrainingResult(
            theta=theta,
            history=history,
            x_data=x_data,
            y_data=y_data,
            x_mean=x_mean,
            x_std=x_std
        )


class UploadDataUseCase:
    def __init__(self):
        self.current_dataset = None
    
    def execute(self, file_content) -> Dataset:
        import pandas as pd
        
        df = pd.read_csv(file_content)
        
        self.current_dataset = Dataset(
            columns=list(df.columns),
            data={col: df[col].tolist() for col in df.columns},
            rows=len(df)
        )
        
        return self.current_dataset
    
    def get_current(self) -> Dataset:
        return self.current_dataset