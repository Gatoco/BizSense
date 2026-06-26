from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import asdict

from ...application.use_cases import UploadDataUseCase, TrainModelUseCase


app = FastAPI(title="BizSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

upload_use_case = UploadDataUseCase()
train_use_case = TrainModelUseCase()


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    dataset = upload_use_case.execute(file.file)
    
    return {
        'columns': dataset.columns,
        'preview': [
            {col: dataset.data[col][i] for col in dataset.columns}
            for i in range(min(10, dataset.rows))
        ],
        'rows': dataset.rows
    }


@app.post("/train")
async def train(params: dict):
    dataset = upload_use_case.get_current()
    
    result = train_use_case.execute(
        dataset=dataset,
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


@app.post("/predict")
async def predict(params: dict):
    import numpy as np
    
    x_values = np.array(params['x_values'])
    theta = np.array(params['theta'])
    predictions = theta[0] + theta[1] * x_values
    
    return {'predictions': predictions.tolist()}