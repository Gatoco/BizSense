from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dataclasses import asdict

from ml.application.use_cases import (
    UploadDataUseCase, ListDatasetsUseCase, GetDatasetUseCase, DeleteDatasetUseCase,
    TrainModelUseCase, GetTrainingHistoryUseCase, GetStatsUseCase,
    ManageConfigUseCase, PredictUseCase, GenerateInsightsUseCase,
    DetectAIProviderUseCase, StartModelUseCase, ChatUseCase, ChatHistoryUseCase
)
from ml.application.ports.dataset_repository import DatasetRepository
from ml.application.ports.training_repository import TrainingRepository
from ml.application.ports.config_repository import ConfigRepository
from ml.infrastructure.storage.sqlite_adapter import (
    SQLiteDatasetRepository, SQLiteTrainingRepository, SQLiteConfigRepository,
    SQLiteChatRepository, init_db
)
from ml.infrastructure.ai.ollama_adapter import OllamaAdapter
from ml.infrastructure.ai.lmstudio_adapter import LMStudioAdapter
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
insights_use_case = GenerateInsightsUseCase(training_repo, dataset_repo)
chat_repo = SQLiteChatRepository()
chat_history_use_case = ChatHistoryUseCase(chat_repo)

ollama_adapter = OllamaAdapter()
lmstudio_adapter = LMStudioAdapter()
detect_provider_use_case = DetectAIProviderUseCase([ollama_adapter, lmstudio_adapter])


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


@app.get("/insights")
async def insights():
    result = insights_use_case.execute()
    if not result:
        return {'has_data': False}
    return {
        'has_data': True,
        'trend_direction': result.trend_direction,
        'trend_rate': result.trend_rate,
        'trend_rate_label': result.trend_rate_label,
        'prediction_next': result.prediction_next,
        'confidence': result.confidence,
        'confidence_score': result.confidence_score,
        'interpretation': result.interpretation,
        'model_type': result.model_type,
        'dataset_name': result.dataset_name,
        'dataset_rows': result.dataset_rows,
        'equation': result.equation,
        'x_col': result.x_col,
        'y_col': result.y_col,
        'x_data': result.x_data,
        'y_data': result.y_data,
        'predictions': result.predictions,
        'future_predictions': result.future_predictions,
        'future_x': result.future_x
    }


@app.get("/ai/status")
async def ai_status():
    config = manage_config_use_case.get()
    provider_name = config.ai_provider

    if provider_name == "auto":
        detected = detect_provider_use_case.execute()
        if detected:
            provider = detect_provider_use_case.get_provider(detected)
            return {
                'available': True,
                'provider': detected,
                'model': config.ai_model,
                'models': provider.list_models()
            }
        return {
            'available': False,
            'provider': None,
            'model': config.ai_model,
            'models': []
        }
    else:
        provider = detect_provider_use_case.get_provider(provider_name)
        if provider and provider.is_available():
            return {
                'available': True,
                'provider': provider_name,
                'model': config.ai_model,
                'models': provider.list_models()
            }
        return {
            'available': False,
            'provider': provider_name,
            'model': config.ai_model,
            'models': []
        }


@app.get("/ai/models")
async def ai_models():
    config = manage_config_use_case.get()
    provider_name = config.ai_provider

    if provider_name == "auto":
        provider = detect_provider_use_case.get_auto()
    else:
        provider = detect_provider_use_case.get_provider(provider_name)

    if not provider:
        return {'models': []}

    return {'models': provider.list_models()}


@app.post("/ai/start")
async def ai_start():
    config = manage_config_use_case.get()
    provider_name = config.ai_provider

    if provider_name == "auto":
        detected = detect_provider_use_case.execute()
        if detected:
            provider = detect_provider_use_case.get_provider(detected)
        else:
            provider = ollama_adapter
    else:
        provider = detect_provider_use_case.get_provider(provider_name)
        if not provider:
            provider = ollama_adapter

    use_case = StartModelUseCase(provider)
    started = use_case.execute()

    if started and provider.get_name() == "ollama":
        models = provider.list_models()
        if config.ai_model not in models:
            provider.pull_model(config.ai_model)

    return {
        'started': started,
        'provider': provider.get_name(),
        'models': provider.list_models() if started else []
    }


@app.post("/ai/chat")
async def ai_chat(params: dict):
    config = manage_config_use_case.get()
    provider_name = config.ai_provider

    if provider_name == "auto":
        provider = detect_provider_use_case.get_auto()
    else:
        provider = detect_provider_use_case.get_provider(provider_name)

    if not provider or not provider.is_available():
        raise HTTPException(status_code=503, detail="No AI provider available. Start Ollama or LM Studio.")

    insights_result = insights_use_case.execute()
    if not insights_result:
        raise HTTPException(status_code=400, detail="No training data available. Train a model first.")

    messages = params.get('messages', [])
    model = params.get('model', config.ai_model)

    chat_use_case = ChatUseCase(provider, chat_repo)

    def stream():
        collected = []
        try:
            chat_use_case.execute(
                messages=messages,
                model=model,
                insights=insights_result,
                on_token=lambda token: collected.append(token)
            )
            full = "".join(collected)
            yield f"data: {__import__('json').dumps({'token': full, 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {__import__('json').dumps({'error': str(e)})}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/chat/history")
async def chat_history():
    messages = chat_history_use_case.list()
    return [
        {
            'id': m.id,
            'role': m.role,
            'content': m.content,
            'training_id': m.training_id,
            'created_at': m.created_at
        }
        for m in messages
    ]


@app.delete("/chat/history")
async def clear_chat_history():
    chat_history_use_case.clear()
    return {'message': 'Chat history cleared'}