from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict

from ml.application.use_cases import (
    UploadDataUseCase, ListDatasetsUseCase, GetDatasetUseCase, DeleteDatasetUseCase,
    TrainModelUseCase, GetTrainingHistoryUseCase, GetStatsUseCase,
    ManageConfigUseCase, PredictUseCase
)
from ml.application.ports.dataset_repository import DatasetRepository
from ml.application.ports.training_repository import TrainingRepository
from ml.application.ports.config_repository import ConfigRepository
from ml.infrastructure.storage.sqlite_adapter import (
    SQLiteDatasetRepository, SQLiteTrainingRepository, SQLiteConfigRepository, init_db
)
from ml.domain.models import AppConfig


app = FastAPI(title="BizSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()

dataset_repo: DatasetRepository = SQLiteDatasetRepository()
training_repo: TrainingRepository = SQLiteTrainingRepository()
config_repo: ConfigRepository = SQLiteConfigRepository()

upload_use_case = UploadDataUseCase(dataset_repo)
list_datasets_use_case = ListDatasetsUseCase(dataset_repo)
get_dataset_use_case = GetDatasetUseCase(dataset_repo)
delete_dataset_use_case = DeleteDatasetUseCase(dataset_repo)
train_use_case = TrainModelUseCase(dataset_repo, training_repo)
get_trainings_use_case = GetTrainingHistoryUseCase(training_repo)
get_stats_use_case = GetStatsUseCase(dataset_repo, training_repo)
manage_config_use_case = ManageConfigUseCase(config_repo)
predict_use_case = PredictUseCase()


@app.get("/stats")
async def stats():
    s = get_stats_use_case.execute()
    result = {
        'datasets_count': s.datasets_count,
        'trainings_count': s.trainings_count,
        'models_implemented': s.models_implemented,
        'last_training': None
    }
    if s.last_training:
        result['last_training'] = {
            'id': s.last_training.id,
            'dataset_name': s.last_training.dataset_name,
            'model_type': s.last_training.model_type,
            'equation': s.last_training.equation,
            'final_cost': s.last_training.final_cost,
            'created_at': s.last_training.created_at
        }
    return result


@app.get("/datasets")
async def list_datasets():
    datasets = list_datasets_use_case.execute()
    return [
        {
            'id': ds.id,
            'name': ds.name,
            'filename': ds.filename,
            'columns': ds.columns,
            'rows': ds.rows,
            'created_at': ds.created_at
        }
        for ds in datasets
    ]


@app.post("/datasets")
async def upload_dataset(file: UploadFile = File(...)):
    import os
    filename = file.filename
    ds = upload_use_case.execute(filename, file.file)

    import pandas as pd
    from ml.infrastructure.storage.sqlite_adapter import DATASETS_DIR
    filepath = os.path.join(DATASETS_DIR, ds.filename)
    df = pd.read_csv(filepath)
    preview = df.head(10).to_dict(orient='records')

    return {
        'id': ds.id,
        'name': ds.name,
        'columns': ds.columns,
        'rows': ds.rows,
        'preview': preview,
        'created_at': ds.created_at
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: int, load_data: bool = False):
    try:
        ds = get_dataset_use_case.execute(dataset_id, load_data=load_data)

        import pandas as pd
        from ml.infrastructure.storage.sqlite_adapter import DATASETS_DIR
        import os
        filepath = os.path.join(DATASETS_DIR, ds.filename)
        df = pd.read_csv(filepath)
        preview = df.head(10).to_dict(orient='records')

        return {
            'id': ds.id,
            'name': ds.name,
            'columns': ds.columns,
            'rows': ds.rows,
            'preview': preview,
            'created_at': ds.created_at
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: int):
    try:
        delete_dataset_use_case.execute(dataset_id)
        return {'message': 'Dataset deleted'}
    except ValueError:
        raise HTTPException(status_code=404, detail="Dataset not found")


@app.post("/train")
async def train(params: dict):
    try:
        result = train_use_case.execute(
            dataset_id=params['dataset_id'],
            x_col=params['x_col'],
            y_col=params['y_col'],
            alpha=params.get('alpha', 0.01),
            iterations=params.get('iterations', 100)
        )

        return {
            'theta': result.theta,
            'history': [asdict(step) for step in result.history],
            'x_data': result.x_data,
            'y_data': result.y_data,
            'x_norm': {'mean': result.x_mean, 'std': result.x_std}
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/trainings")
async def list_trainings():
    trainings = get_trainings_use_case.execute()
    return [
        {
            'id': t.id,
            'dataset_id': t.dataset_id,
            'dataset_name': t.dataset_name,
            'model_type': t.model_type,
            'x_col': t.x_col,
            'y_col': t.y_col,
            'alpha': t.alpha,
            'iterations': t.iterations,
            'theta_0': t.theta_0,
            'theta_1': t.theta_1,
            'final_cost': t.final_cost,
            'equation': t.equation,
            'created_at': t.created_at
        }
        for t in trainings
    ]


@app.get("/trainings/{training_id}")
async def get_training(training_id: int):
    from ml.infrastructure.storage.sqlite_adapter import SQLiteTrainingRepository
    repo = SQLiteTrainingRepository()
    try:
        t = repo.get_by_id(training_id)
        return {
            'id': t.id,
            'dataset_id': t.dataset_id,
            'dataset_name': t.dataset_name,
            'model_type': t.model_type,
            'x_col': t.x_col,
            'y_col': t.y_col,
            'alpha': t.alpha,
            'iterations': t.iterations,
            'theta_0': t.theta_0,
            'theta_1': t.theta_1,
            'final_cost': t.final_cost,
            'equation': t.equation,
            'created_at': t.created_at
        }
    except ValueError:
        raise HTTPException(status_code=404, detail="Training not found")


@app.get("/config")
async def get_config():
    config = manage_config_use_case.get()
    return config.to_dict()


@app.post("/config")
async def update_config(params: dict):
    config = AppConfig.from_dict(params)
    manage_config_use_case.update(config)
    return {'message': 'Config updated'}


@app.post("/predict")
async def predict(params: dict):
    predictions = predict_use_case.execute(
        x_values=params['x_values'],
        theta=params['theta']
    )
    return {'predictions': predictions}