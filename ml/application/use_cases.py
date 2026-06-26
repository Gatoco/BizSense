from datetime import datetime
import pandas as pd
import numpy as np
from ml.domain.models import Dataset, Training, TrainingResult, AppConfig, Stats, InsightsMetrics, ChatMessage
from ml.domain.services import gradient_descent
from ml.application.ports.dataset_repository import DatasetRepository
from ml.application.ports.training_repository import TrainingRepository
from ml.application.ports.config_repository import ConfigRepository
from ml.application.ports.ai_provider import AIProvider


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


class GenerateInsightsUseCase:
    def __init__(self, training_repo: TrainingRepository, dataset_repo: DatasetRepository):
        self.training_repo = training_repo
        self.dataset_repo = dataset_repo

    def execute(self) -> InsightsMetrics | None:
        training = self.training_repo.get_last()
        if not training:
            return None

        dataset = self.dataset_repo.get_by_id(training.dataset_id, load_data=True)

        x_data = dataset.get_column(training.x_col)
        y_data = dataset.get_column(training.y_col)

        theta_0 = training.theta_0
        theta_1 = training.theta_1

        x_arr = np.array(x_data, dtype=float)
        y_arr = np.array(y_data, dtype=float)

        y_pred = theta_0 + theta_1 * x_arr
        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        if theta_1 > 0:
            trend_direction = "creciente"
        elif theta_1 < 0:
            trend_direction = "decreciente"
        else:
            trend_direction = "estable"

        trend_rate = abs(theta_1)
        unit = training.y_col.lower() if training.y_col else "unidad"
        trend_rate_label = f"${trend_rate:.0f}/{training.x_col}" if "venta" in unit or "precio" in unit else f"{trend_rate:.1f}/{training.x_col}"

        x_max = max(x_data)
        prediction_next = theta_0 + theta_1 * (x_max + 1)

        if r_squared > 0.8:
            confidence = "alta"
        elif r_squared > 0.5:
            confidence = "media"
        else:
            confidence = "baja"

        if trend_direction == "creciente":
            interpretation = f"Tus {training.y_col} suben aproximadamente {trend_rate_label}. Si continua esta tendencia, el proximo {training.x_col} esperarias alrededor de ${prediction_next:,.0f}."
        elif trend_direction == "decreciente":
            interpretation = f"Tus {training.y_col} bajan aproximadamente {trend_rate_label}. Si continua esta tendencia, el proximo {training.x_col} esperarias alrededor de ${prediction_next:,.0f}. Te recomendamos revisar estrategias para revertir esta tendencia."
        else:
            interpretation = f"Tus {training.y_col} se mantienen estables. El proximo {training.x_col} esperarias alrededor de ${prediction_next:,.0f}."

        future_x = [x_max + i for i in range(1, 6)]
        future_predictions = [theta_0 + theta_1 * x for x in future_x]

        predictions = [theta_0 + theta_1 * x for x in x_data]

        return InsightsMetrics(
            trend_direction=trend_direction,
            trend_rate=trend_rate,
            trend_rate_label=trend_rate_label,
            prediction_next=prediction_next,
            confidence=confidence,
            confidence_score=float(r_squared),
            interpretation=interpretation,
            model_type=training.model_type,
            dataset_name=training.dataset_name,
            dataset_rows=len(x_data),
            equation=training.equation,
            x_col=training.x_col,
            y_col=training.y_col,
            x_data=x_data,
            y_data=y_data,
            predictions=predictions,
            future_predictions=future_predictions,
            future_x=future_x
        )


class DetectAIProviderUseCase:
    def __init__(self, providers: list):
        self.providers = providers

    def execute(self) -> str | None:
        for p in self.providers:
            if p.is_available():
                return p.get_name()
        return None

    def get_provider(self, name: str) -> AIProvider | None:
        for p in self.providers:
            if p.get_name() == name:
                return p
        return None

    def get_auto(self) -> AIProvider | None:
        for p in self.providers:
            if p.is_available():
                return p
        return None


class StartModelUseCase:
    def __init__(self, provider: AIProvider):
        self.provider = provider

    def execute(self) -> bool:
        if not self.provider.is_available():
            if not self.provider.start_server():
                return False
        return self.provider.is_available()


class ChatUseCase:
    SYSTEM_PROMPT = """Eres un asistente de analisis de negocios para Pymes.
Estas ayudando a un trabajador de una pyme a entender los resultados
de un modelo de Machine Learning.

RESULTADOS DEL ANALISIS:
- Modelo: {model_type}
- Dataset: {dataset_name} ({rows} registros)
- Tendencia: {trend_direction} ({trend_rate_label})
- Prediccion proximo periodo: ${prediction_next:,.0f}
- Confianza del modelo: {confidence} (R² = {confidence_score:.2f})
- Ecuacion: {equation}

INSTRUCCIONES:
1. Responde en espanol, lenguaje simple, sin terminos tecnicos
2. Usa los resultados reales para dar recomendaciones
3. Relaciona con decisiones de negocio: inventario, precios, personal
4. Si no hay datos suficientes, se honesto
5. No inventes numeros, usa solo los proporcionados
6. Se conciso y directo"""

    def __init__(self, provider: AIProvider, chat_repo=None):
        self.provider = provider
        self.chat_repo = chat_repo

    def build_system_prompt(self, insights: InsightsMetrics) -> str:
        return self.SYSTEM_PROMPT.format(
            model_type=insights.model_type,
            dataset_name=insights.dataset_name,
            rows=insights.dataset_rows,
            trend_direction=insights.trend_direction,
            trend_rate_label=insights.trend_rate_label,
            prediction_next=insights.prediction_next,
            confidence=insights.confidence,
            confidence_score=insights.confidence_score,
            equation=insights.equation
        )

    def execute(self, messages: list, model: str, insights: InsightsMetrics, on_token=None) -> str:
        system_prompt = self.build_system_prompt(insights)

        full_messages = [{"role": "system", "content": system_prompt}] + messages

        response = self.provider.chat(full_messages, model, on_token=on_token)

        if self.chat_repo:
            from datetime import datetime
            now = datetime.now().isoformat(timespec='seconds')
            for msg in messages[-2:]:
                self.chat_repo.save(ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    created_at=now
                ))
            self.chat_repo.save(ChatMessage(
                role="assistant",
                content=response,
                created_at=now
            ))

        return response


class ChatHistoryUseCase:
    def __init__(self, chat_repo):
        self.chat_repo = chat_repo

    def list(self, training_id: int = None) -> list:
        return self.chat_repo.list_all(training_id)

    def clear(self, training_id: int = None) -> None:
        self.chat_repo.clear(training_id)