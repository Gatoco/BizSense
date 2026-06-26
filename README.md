# BizSense - Sistema de Analisis Predictivo para Pymes

## Descripcion

BizSense es una aplicacion de escritorio que ayuda a pequenas y medianas empresas (pymes) a tomar decisiones basadas en datos. Permite predecir ventas, analizar retencion de clientes y optimizar precios mediante modelos de Machine Learning, aplicando conceptos de calculo diferencial (derivadas, limites y optimizacion).

## Integrantes

- Benjamin Cifuentes
- Benjamin Contesso

## Asignatura

Calculo Diferencial

## Instalacion y Ejecucion

### Requisitos
- Python 3.8+
- Node.js 18+
- npm

### Ejecutar

```bash
./start.sh
```

Este comando:
1. Crea y activa el entorno virtual de Python
2. Instala todas las dependencias
3. Inicia el backend (FastAPI) en puerto 5000
4. Inicia la aplicacion Electron

## Arquitectura

El proyecto sigue una **arquitectura hexagonal** (puertos y adaptadores):

```
BizSense/
├── app/                                # Frontend (Electron)
│   ├── main.js                         # Electron main process
│   ├── preload.js                      # Bridge seguro
│   └── renderer/
│       ├── index.html                  # Dashboard (pagina inicial)
│       ├── models.html                 # Modelos ML
│       ├── graphics.html               # Entrenamiento animado
│       ├── data.html                   # Gestion de CSVs
│       ├── config.html                 # Configuracion
│       ├── style.css                   # Tema oscuro (LM Studio style)
│       └── js/
│           ├── sidebar.js              # Sidebar compartido (inyectado dinamicamente)
│           ├── dashboard.js            # Logica dashboard
│           ├── models.js               # Logica modelos
│           ├── graphics.js             # Logica entrenamiento + animacion
│           ├── data.js                 # Logica gestion CSVs
│           └── config.js               # Logica configuracion
│
├── ml/                                 # Backend (Python)
│   ├── __init__.py
│   ├── main.py                         # Punto de entrada FastAPI
│   │
│   ├── domain/                         # CAPA INTERNA (logica pura)
│   │   ├── models.py                   # Entidades: Dataset, Training, Config, Stats
│   │   └── services.py                 # Gradiente descendente
│   │
│   ├── application/                    # CAPA MEDIA (casos de uso)
│   │   ├── use_cases.py                # 9 casos de uso
│   │   └── ports/                      # PUERTOS (interfaces abstractas)
│   │       ├── dataset_repository.py   # save, get, list, delete
│   │       ├── training_repository.py  # save, get, list, get_last
│   │       └── config_repository.py    # get_all, update
│   │
│   └── infrastructure/                 # CAPA EXTERNA (adaptadores)
│       ├── api/
│       │   └── fastapi_adapter.py      # 11 endpoints HTTP
│       └── storage/
│           └── sqlite_adapter.py       # Implementa los 3 puertos con SQLite
│
├── data/
│   ├── bizsense.db                     # SQLite (datasets, entrenamientos, config)
│   ├── datasets/                       # CSVs subidos
│   └── sales_sample.csv                # Dataset de ejemplo
│
├── start.sh                            # Script de inicio
├── package.json
└── requirements.txt
```

### Flujo de dependencias (hacia adentro)

```
infrastructure/ ──> application/ ──> domain/
  (adaptadores)     (casos de uso)   (logica pura)
```

- **domain/** no depende de nada
- **application/** depende solo de domain
- **infrastructure/** implementa los puertos de application

### Puertos (interfaces)

```python
class DatasetRepository(ABC):      # save, get_by_id, list_all, delete
class TrainingRepository(ABC):     # save, get_by_id, list_all, get_last
class ConfigRepository(ABC):       # get_all, update
```

### Adaptadores

| Adaptador | Tipo | Tecnologia |
|-----------|------|------------|
| `fastapi_adapter.py` | Driving (entrada) | FastAPI |
| `sqlite_adapter.py` | Driven (salida) | sqlite3 (stdlib) |

### Casos de uso

| Caso de uso | Responsabilidad |
|-------------|-----------------|
| `UploadDataUseCase` | Parsear CSV, guardar en BD + disco |
| `ListDatasetsUseCase` | Listar datasets |
| `GetDatasetUseCase` | Obtener dataset por ID |
| `DeleteDatasetUseCase` | Eliminar dataset |
| `TrainModelUseCase` | Ejecutar gradiente descendente, guardar resultado |
| `GetTrainingHistoryUseCase` | Listar entrenamientos |
| `GetStatsUseCase` | Metricas para dashboard |
| `ManageConfigUseCase` | Leer/actualizar configuracion |
| `PredictUseCase` | Predecir con modelo entrenado |

## Endpoints del backend

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/stats` | GET | Metricas del dashboard |
| `/datasets` | GET | Lista de datasets |
| `/datasets` | POST | Subir nuevo CSV |
| `/datasets/{id}` | GET | Preview de un dataset |
| `/datasets/{id}` | DELETE | Eliminar dataset |
| `/train` | POST | Entrenar modelo |
| `/trainings` | GET | Historial de entrenamientos |
| `/trainings/{id}` | GET | Detalle de un entrenamiento |
| `/config` | GET | Configuracion actual |
| `/config` | POST | Actualizar configuracion |
| `/predict` | POST | Prediccion con modelo entrenado |

## Paginas

| Pagina | Contenido |
|--------|-----------|
| **Dashboard** | Metricas + ultimo entrenamiento + acceso rapido |
| **Modelos ML** | Lista de modelos con estado, descripcion y formulas |
| **Funcion Grafica** | Seleccionar CSV + configurar + entrenar + animacion paso a paso |
| **Datos** | Tabla de CSVs + preview + upload + eliminar |
| **Configuracion** | Alpha default, iteraciones default, tema, idioma |

## Modelos de Machine Learning

### 1. Regresion Lineal (Implementado)
- Uso: Prediccion de ventas futuras
- Derivadas: Descenso por gradiente para minimizar el error
- Visualizacion: Animacion paso a paso del entrenamiento

### 2. Regresion Logistica (Proximamente)
- Uso: Clasificacion (ventas buenas/malas, clientes que se van/que se quedan)
- Derivadas: Funcion sigmoide y su derivada

### 3. K-Means (Proximamente)
- Uso: Clustering (segmentar negocios o productos)
- Limites: Convergencia cuando n tiende a infinito

### 4. Red Neuronal (Proximamente)
- Uso: Detectar patrones complejos
- Derivadas: Backpropagation (regla de la cadena)

## Calculo Diferencial en el Proyecto

### Derivadas
- Descenso por gradiente: minimos de la funcion de costo
- Optimizacion: encontrar maximos y minimos
- Regla de la cadena: backpropagation en redes neuronales

### Limites
- Convergencia de algoritmos iterativos
- Comportamiento asintotico de funciones
- Proyecciones a largo plazo

## Tecnologias

| Capa | Herramienta |
|------|-------------|
| Frontend | Electron + HTML/CSS/JS + Chart.js + MathJax |
| Backend | FastAPI (Python) |
| ML | NumPy + Pandas |
| Storage | SQLite (stdlib) |
| Arquitectura | Hexagonal (Puertos y Adaptadores) |

## Licencia

MIT