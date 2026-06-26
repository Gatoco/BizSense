import sqlite3
import os
import json
import pandas as pd
from ml.domain.models import Dataset, Training, AppConfig, ChatMessage
from ml.application.ports.dataset_repository import DatasetRepository
from ml.application.ports.training_repository import TrainingRepository
from ml.application.ports.config_repository import ConfigRepository


DB_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'bizsense.db')
DATASETS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'datasets')


class SQLiteDatasetRepository(DatasetRepository):
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs(DATASETS_DIR, exist_ok=True)

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def save(self, dataset: Dataset) -> int:
        filepath = os.path.join(DATASETS_DIR, dataset.filename)
        df = pd.DataFrame(dataset._data)
        df.to_csv(filepath, index=False)

        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO datasets (name, filename, columns, rows, created_at) VALUES (?, ?, ?, ?, ?)',
                (dataset.name, dataset.filename, json.dumps(dataset.columns), dataset.rows, dataset.created_at)
            )
            conn.commit()
            return cursor.lastrowid

    def get_by_id(self, dataset_id: int, load_data: bool = False) -> Dataset:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM datasets WHERE id = ?', (dataset_id,))
            row = cursor.fetchone()

        if not row:
            raise ValueError(f"Dataset {dataset_id} not found")

        ds = Dataset(
            id=row[0], name=row[1], filename=row[2],
            columns=json.loads(row[3]), rows=row[4], created_at=row[5]
        )

        if load_data:
            filepath = os.path.join(DATASETS_DIR, ds.filename)
            df = pd.read_csv(filepath)
            ds.set_data({col: df[col].tolist() for col in df.columns})

        return ds

    def list_all(self) -> list[Dataset]:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM datasets ORDER BY created_at DESC')
            rows = cursor.fetchall()

        return [
            Dataset(id=r[0], name=r[1], filename=r[2], columns=json.loads(r[3]), rows=r[4], created_at=r[5])
            for r in rows
        ]

    def delete(self, dataset_id: int) -> None:
        ds = self.get_by_id(dataset_id)
        filepath = os.path.join(DATASETS_DIR, ds.filename)
        if os.path.exists(filepath):
            os.remove(filepath)

        with self._conn() as conn:
            conn.execute('DELETE FROM datasets WHERE id = ?', (dataset_id,))
            conn.commit()


class SQLiteTrainingRepository(TrainingRepository):
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def save(self, training: Training) -> int:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO trainings
                (dataset_id, dataset_name, model_type, x_col, y_col, alpha, iterations, theta_0, theta_1, final_cost, equation, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    training.dataset_id, training.dataset_name, training.model_type,
                    training.x_col, training.y_col, training.alpha, training.iterations,
                    training.theta_0, training.theta_1, training.final_cost,
                    training.equation, training.created_at
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_by_id(self, training_id: int) -> Training:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trainings WHERE id = ?', (training_id,))
            row = cursor.fetchone()

        if not row:
            raise ValueError(f"Training {training_id} not found")

        return Training(
            id=row[0], dataset_id=row[1], dataset_name=row[2], model_type=row[3],
            x_col=row[4], y_col=row[5], alpha=row[6], iterations=row[7],
            theta_0=row[8], theta_1=row[9], final_cost=row[10],
            equation=row[11], created_at=row[12]
        )

    def list_all(self) -> list[Training]:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trainings ORDER BY created_at DESC')
            rows = cursor.fetchall()

        return [
            Training(
                id=r[0], dataset_id=r[1], dataset_name=r[2], model_type=r[3],
                x_col=r[4], y_col=r[5], alpha=r[6], iterations=r[7],
                theta_0=r[8], theta_1=r[9], final_cost=r[10],
                equation=r[11], created_at=r[12]
            )
            for r in rows
        ]

    def get_last(self) -> Training | None:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM trainings ORDER BY created_at DESC LIMIT 1')
            row = cursor.fetchone()

        if not row:
            return None

        return Training(
            id=row[0], dataset_id=row[1], dataset_name=row[2], model_type=row[3],
            x_col=row[4], y_col=row[5], alpha=row[6], iterations=row[7],
            theta_0=row[8], theta_1=row[9], final_cost=row[10],
            equation=row[11], created_at=row[12]
        )


class SQLiteConfigRepository(ConfigRepository):
    DEFAULTS = {
        'default_alpha': '0.01',
        'default_iterations': '100',
        'theme': 'dark',
        'language': 'es',
        'ai_provider': 'auto',
        'ai_model': 'qwen2.5:1.5b',
        'ai_endpoint': ''
    }

    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def get_all(self) -> AppConfig:
        config_dict = dict(self.DEFAULTS)
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT key, value FROM config')
            for key, value in cursor.fetchall():
                config_dict[key] = value

        return AppConfig.from_dict(config_dict)

    def update(self, config: AppConfig) -> None:
        data = config.to_dict()
        with self._conn() as conn:
            for key, value in data.items():
                conn.execute(
                    'INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)',
                    (key, str(value))
                )
            conn.commit()


def init_db(db_path: str = None):
    db_path = db_path or DB_PATH
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                filename TEXT NOT NULL,
                columns TEXT NOT NULL,
                rows INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trainings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                dataset_name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                x_col TEXT NOT NULL,
                y_col TEXT NOT NULL,
                alpha REAL NOT NULL,
                iterations INTEGER NOT NULL,
                theta_0 REAL NOT NULL,
                theta_1 REAL NOT NULL,
                final_cost REAL NOT NULL,
                equation TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                training_id INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()


class SQLiteChatRepository:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def save(self, msg: ChatMessage) -> int:
        with self._conn() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO chat_history (role, content, training_id, created_at) VALUES (?, ?, ?, ?)',
                (msg.role, msg.content, msg.training_id, msg.created_at)
            )
            conn.commit()
            return cursor.lastrowid

    def list_all(self, training_id: int = None) -> list:
        with self._conn() as conn:
            cursor = conn.cursor()
            if training_id:
                cursor.execute('SELECT * FROM chat_history WHERE training_id = ? ORDER BY id ASC', (training_id,))
            else:
                cursor.execute('SELECT * FROM chat_history ORDER BY id ASC')
            return [
                ChatMessage(id=r[0], role=r[1], content=r[2], training_id=r[3], created_at=r[4])
                for r in cursor.fetchall()
            ]

    def clear(self, training_id: int = None) -> None:
        with self._conn() as conn:
            if training_id:
                conn.execute('DELETE FROM chat_history WHERE training_id = ?', (training_id,))
            else:
                conn.execute('DELETE FROM chat_history')
            conn.commit()