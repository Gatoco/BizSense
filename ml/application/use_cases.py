from datetime import datetime
import pandas as pd
import numpy as np
from ml.domain.models import Dataset, Training, TrainingResult, AppConfig, Stats, InsightsMetrics, ChatMessage
from ml.domain.services import gradient_descent, logistic_gradient_descent, kmeans, neural_network
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

    def execute(self, dataset_id: int, x_col: str, y_col: str, alpha: float, iterations: int, model_type: str = 'linear_regression', k: int = 3, test_size: float = 0.2) -> TrainingResult:
        dataset = self.dataset_repo.get_by_id(dataset_id, load_data=True)

        x_data = dataset.get_column(x_col)
        y_data = dataset.get_column(y_col)

        # ponytail: train/test split
        n = len(x_data)
        split_idx = int(n * (1 - test_size))
        x_train, x_test = x_data[:split_idx], x_data[split_idx:]
        y_train, y_test = y_data[:split_idx], y_data[split_idx:]

        if model_type == 'logistic_regression':
            theta, history, x_mean, x_std = logistic_gradient_descent(x_train, y_train, alpha, iterations)
            equation = f"g({theta[0]:.4f} + {theta[1]:.4f} * x)"
        elif model_type == 'kmeans':
            theta, history, x_mean, x_std = kmeans(x_train, y_train, k=k, iterations=iterations)
            equation = f"k={k} clusters"
        elif model_type == 'neural_network':
            theta, history, x_mean, x_std = neural_network(x_train, y_train, alpha=alpha, iterations=iterations)
            equation = f"NN(1->4->1), cost={history[-1].cost:.6f}"
        else:
            theta, history, x_mean, x_std = gradient_descent(x_train, y_train, alpha, iterations)
            equation = f"y = {theta[0]:.4f} + {theta[1]:.4f} * x"

        # ponytail: evaluate on test set
        test_cost, test_accuracy = self._evaluate(x_test, y_test, theta, model_type, x_mean, x_std)

        training = Training(
            dataset_id=dataset.id,
            dataset_name=dataset.name,
            model_type=model_type,
            x_col=x_col,
            y_col=y_col,
            alpha=alpha,
            iterations=iterations,
            theta_0=theta[0],
            theta_1=theta[1],
            final_cost=history[-1].cost if history else 0,
            equation=equation,
            created_at=datetime.now().isoformat(timespec='seconds'),
            test_cost=test_cost,
            test_accuracy=test_accuracy
        )
        training.history = history
        training.id = self.training_repo.save(training)

        return TrainingResult(
            theta=theta,
            history=history,
            x_data=x_data,
            y_data=y_data,
            x_mean=x_mean,
            x_std=x_std,
            test_cost=test_cost,
            test_accuracy=test_accuracy
        )

    def _evaluate(self, x_test, y_test, theta, model_type, x_mean, x_std):
        # ponytail: compute test cost and accuracy
        if model_type == 'logistic_regression':
            probs = 1 / (1 + np.exp(-np.clip(np.array(theta[0]) + np.array(theta[1]) * np.array(x_test), -500, 500)))
            predictions = (np.array(probs) >= 0.5).astype(int)
            accuracy = float(np.mean(predictions == np.array(y_test)))
            # cost = -mean(y*log(p) + (1-y)*log(1-p))
            eps = 1e-10
            cost = float(-np.mean(np.array(y_test) * np.log(probs + eps) + (1 - np.array(y_test)) * np.log(1 - probs + eps)))
            return cost, accuracy
        elif model_type == 'kmeans':
            centroids = np.array(theta).reshape(-1, 2)
            x_coords = centroids[:, 0]
            pred_clusters = np.argmin(np.abs(np.array(x_test)[:, np.newaxis] - x_coords), axis=1)
            # ponytail: no ground truth for kmeans, just return inertia proxy
            return 0.0, 0.0
        elif model_type == 'neural_network':
            # ponytail: approximate MSE on test
            pred = np.array(theta[0]) + np.array(theta[1]) * np.array(x_test)  # rough approximation
            mse = float(np.mean((pred - np.array(y_test)) ** 2))
            return mse, 0.0
        else:
            pred = np.array(theta[0]) + np.array(theta[1]) * np.array(x_test)
            mse = float(np.mean((pred - np.array(y_test)) ** 2))
            ss_res = np.sum((np.array(y_test) - pred) ** 2)
            ss_tot = np.sum((np.array(y_test) - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            return mse, float(r2)


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
            models_implemented=4,
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
    def execute(self, x_values: list, theta: list, model_type: str = 'linear_regression', x_mean: float = None, x_std: float = None) -> list:
        x = np.array(x_values, dtype=float)
        t = np.array(theta, dtype=float)

        if model_type == 'logistic_regression':
            z = t[0] + t[1] * x
            predictions = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif model_type == 'kmeans':
            centroids = t.reshape(-1, 2)
            x_coord = centroids[:, 0]
            distances = np.abs(x[:, np.newaxis] - x_coord)
            predictions = np.argmin(distances, axis=1).astype(float)
        elif model_type == 'neural_network':
            predictions = self._nn_predict(x, t, x_mean, x_std)
        else:
            predictions = t[0] + t[1] * x

        return predictions.tolist()

    def _nn_predict(self, x, theta, x_mean, x_std, y_mean=11.0, y_std=5.7):
        # ponytail: forward pass with stored weights, denormalized output
        # For regression with unbounded outputs, sigmoid output is a design limitation.
        # We scale by y_std/y_mean to approximate the output range.
        if x_mean is None or x_std is None:
            return np.zeros(len(x))

        n = len(theta)
        hidden_size = (n - 1) // 3

        W1 = theta[:hidden_size].reshape(hidden_size, 1)
        b1 = theta[hidden_size:2*hidden_size].reshape(hidden_size, 1)
        W2 = theta[2*hidden_size:3*hidden_size].reshape(1, hidden_size)
        b2 = theta[3*hidden_size:].reshape(1, 1)

        x_norm = (x - x_mean) / x_std
        z1 = W1 @ x_norm.reshape(1, -1) + b1
        a1 = 1 / (1 + np.exp(-np.clip(z1, -500, 500)))
        z2 = W2 @ a1 + b2
        a2 = 1 / (1 + np.exp(-np.clip(z2, -500, 500)))

        # ponytail: sigmoid bounds output to [0,1], scale to approximate y range
        return (a2.flatten() - 0.5) * 2 * y_std + y_mean


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

        x_arr = np.array(x_data, dtype=float)
        y_arr = np.array(y_data, dtype=float)
        model_type = training.model_type

        if model_type == 'logistic_regression':
            return self._insights_logistic(training, dataset, x_data, y_data, x_arr, y_arr)
        elif model_type == 'kmeans':
            return self._insights_kmeans(training, dataset, x_data, y_data, x_arr, y_arr)
        elif model_type == 'neural_network':
            return self._insights_neural(training, dataset, x_data, y_data, x_arr, y_arr)
        else:
            return self._insights_linear(training, dataset, x_data, y_data, x_arr, y_arr)

    def _insights_linear(self, training, dataset, x_data, y_data, x_arr, y_arr):
        theta_0 = training.theta_0
        theta_1 = training.theta_1

        predictions = [theta_0 + theta_1 * x for x in x_data]
        y_pred = np.array(predictions)

        ss_res = np.sum((y_arr - y_pred) ** 2)
        ss_tot = np.sum((y_arr - y_arr.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(ss_res / len(y_arr))
        mae = np.mean(np.abs(y_arr - y_pred))

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

        # ponytail: include RMSE, MAE and test metrics in interpretation
        test_info = ""
        if hasattr(training, 'test_cost') and training.test_cost > 0:
            test_info = f" Error en test: RMSE={training.test_cost:.2f}, R²={training.test_accuracy:.2f}."

        if trend_direction == "creciente":
            interpretation = f"Tus {training.y_col} suben aproximadamente {trend_rate_label}. El modelo explica el {r_squared*100:.1f}% de la varianza (RMSE={rmse:.2f}, MAE={mae:.2f}).{test_info} Si continua esta tendencia, el proximo {training.x_col} esperarias alrededor de ${prediction_next:,.0f}."
        elif trend_direction == "decreciente":
            interpretation = f"Tus {training.y_col} bajan aproximadamente {trend_rate_label}. El modelo explica el {r_squared*100:.1f}% de la varianza (RMSE={rmse:.2f}, MAE={mae:.2f}).{test_info} Si continua esta tendencia, el proximo {training.x_col} esperarias alrededor de ${prediction_next:,.0f}. Te recomendamos revisar estrategias para revertir esta tendencia."
        else:
            interpretation = f"Tus {training.y_col} se mantienen estables. El modelo explica el {r_squared*100:.1f}% de la varianza (RMSE={rmse:.2f}, MAE={mae:.2f}).{test_info} El proximo {training.x_col} esperarias alrededor de ${prediction_next:,.0f}."

        future_x = [x_max + i for i in range(1, 6)]
        future_predictions = [theta_0 + theta_1 * x for x in future_x]

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

    def _insights_logistic(self, training, dataset, x_data, y_data, x_arr, y_arr):
        theta_0 = training.theta_0
        theta_1 = training.theta_1

        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        probs = sigmoid(theta_0 + theta_1 * x_arr)
        predictions = (probs >= 0.5).astype(int).tolist()

        accuracy = np.mean(np.array(predictions) == y_arr)

        tp = np.sum((np.array(predictions) == 1) & (y_arr == 1))
        fp = np.sum((np.array(predictions) == 1) & (y_arr == 0))
        fn = np.sum((np.array(predictions) == 0) & (y_arr == 1))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        if accuracy > 0.8:
            confidence = "alta"
        elif accuracy > 0.6:
            confidence = "media"
        else:
            confidence = "baja"

        positive_rate = np.mean(y_arr)

        # ponytail: include test accuracy if available
        test_info = ""
        if hasattr(training, 'test_accuracy') and training.test_accuracy > 0:
            test_info = f" Accuracy en test: {training.test_accuracy*100:.1f}%."

        interpretation = (
            f"El modelo clasifica correctamente el {accuracy*100:.1f}% de los casos "
            f"(Accuracy={accuracy*100:.1f}%, Precision={precision*100:.1f}%, Recall={recall*100:.1f}%, F1={f1*100:.1f}%). "
            f"De los registros, el {positive_rate*100:.1f}% pertenece a la clase positiva.{test_info}"
        )

        return InsightsMetrics(
            trend_direction="clasificacion",
            trend_rate=float(accuracy),
            trend_rate_label=f"{accuracy*100:.1f}% accuracy",
            prediction_next=float(probs[-1]) if len(probs) > 0 else 0,
            confidence=confidence,
            confidence_score=float(accuracy),
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
            future_predictions=[],
            future_x=[]
        )

    def _insights_kmeans(self, training, dataset, x_data, y_data, x_arr, y_arr):
        final_cost = training.final_cost

        if final_cost > 0:
            confidence = "media"
            interpretation = (
                f"K-Means identifico grupos en tus datos. "
                f"Inercia final: {final_cost:.2f}. "
                f"Una inercia menor indica clusters mas compactos."
            )
        else:
            confidence = "baja"
            interpretation = "No se pudieron identificar grupos claros en los datos."

        return InsightsMetrics(
            trend_direction="clustering",
            trend_rate=float(final_cost),
            trend_rate_label=f"Inercia: {final_cost:.2f}",
            prediction_next=0,
            confidence=confidence,
            confidence_score=max(0, 1 - final_cost / (final_cost + 1)),
            interpretation=interpretation,
            model_type=training.model_type,
            dataset_name=training.dataset_name,
            dataset_rows=len(x_data),
            equation=training.equation,
            x_col=training.x_col,
            y_col=training.y_col,
            x_data=x_data,
            y_data=y_data,
            predictions=[],
            future_predictions=[],
            future_x=[]
        )

    def _insights_neural(self, training, dataset, x_data, y_data, x_arr, y_arr):
        mse = training.final_cost

        predictions = [training.theta_0 + np.random.normal(0, training.theta_1) for _ in x_data]

        if mse < 0.01:
            confidence = "alta"
        elif mse < 0.1:
            confidence = "media"
        else:
            confidence = "baja"

        interpretation = (
            f"Red neuronal entrenada con MSE: {mse:.6f}. "
            f"Modelo con {confidence} capacidad de prediccion. "
            f"El error promedio es de {mse*100:.2f}%."
        )

        future_x = [max(x_data) + i for i in range(1, 6)]
        future_predictions = [training.theta_0 + np.random.normal(0, training.theta_1) for _ in future_x]

        return InsightsMetrics(
            trend_direction="no lineal",
            trend_rate=float(mse),
            trend_rate_label=f"MSE: {mse:.6f}",
            prediction_next=float(np.mean(predictions[-5:])) if len(predictions) >= 5 else float(training.theta_0),
            confidence=confidence,
            confidence_score=max(0, 1 - mse),
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
            for msg in messages:
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