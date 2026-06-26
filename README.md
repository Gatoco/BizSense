# BizSense - Sistema de Analisis Predictivo para Pymes

Aplicación educativa de escritorio que demuestra cómo los conceptos de **Cálculo Diferencial** (derivadas, gradientes, convergencia) se aplican en Machine Learning real. Desarrollada como caso de estudio para la asignatura **Cálculo Diferencial**.

**Autores:** Benjamin Cifuentes, Benjamin Contesso

---

## Descripción General

BizSense es una herramienta interactiva que permite:

- **Subir datos** (archivos CSV) desde la interfaz
- **Entrenar modelos de ML** con visualización paso a paso
- **Ver en tiempo real** cómo funcionan las derivadas y el descenso por gradiente
- **Predecir** nuevos valores usando modelos entrenados
- **Persistir** datasets, entrenamientos y configuración en SQLite

### Enfoque Académico

Se enfatiza el **significado matemático detrás del aprendizaje automático**:
- 🔴 **Derivadas parciales:** Cálculo del gradiente en cada iteración
- 🟢 **Límites:** Convergencia del algoritmo conforme iteraciones → ∞
- 🔵 **Optimización:** Minimización de la función de costo J(θ)

---

## Instalación y Ejecución

### Requisitos Previos

- **Python 3.8+**
- **Node.js 18+** y npm
- **Git** (recomendado)

### Arranque Rápido

```bash
./start.sh
```

**Qué hace el script:**
1. ✅ Crea entorno virtual Python (`.venv`)
2. ✅ Instala dependencias (FastAPI, pandas, numpy)
3. ✅ Instala dependencias Node (Electron)
4. ✅ Inicia backend FastAPI en `http://localhost:5000`
5. ✅ Inicia interfaz Electron

**Tiempo de inicio:** ~5-10 segundos en primer arranque, ~2-3 segundos después

### Arranque Manual (desarrollo)

```bash
# Terminal 1: Backend
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.main

# Terminal 2: Frontend
npm install
npm start
```

---

## Arquitectura

### Visión General

```
┌─────────────────────────────────────────────────────────────┐
│ ELECTRON (Frontend)                                         │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ Páginas: index.html, data.html, graphics.html, etc.     │ │
│ │ Scripts: dashboard.js, graphics.js, data.js, etc.       │ │
│ │ Comunicación: fetch() → http://localhost:5000           │ │
│ └──────────────────────────────────────────────────────────┘ │
└────────────────────────┬──────────────────────────────────────┘
                         │ HTTP (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ FASTAPI (Backend)                                           │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ 11 Endpoints: /datasets, /train, /stats, etc.         │  │
│ │ Adaptador: fastapi_adapter.py                         │  │
│ └────────────────────────────────────────────────────────┘  │
│          │                                    │             │
│          ▼                                    ▼             │
│ ┌──────────────────┐              ┌─────────────────────┐  │
│ │ USE CASES        │              │ DOMAIN SERVICES     │  │
│ │ (9 casos)        │              │ gradiente_descent() │  │
│ │ UploadDataUseCase│──────────────┤ math formulas       │  │
│ │ TrainModelUseCase│              └─────────────────────┘  │
│ │ etc.             │                                       │
│ └──────────────────┘                                       │
│          │                                                  │
│          ▼                                                  │
│ ┌──────────────────────────────────────────────────────┐  │
│ │ SQLITE ADAPTER (sqlite_adapter.py)                  │  │
│ │ - SQLiteDatasetRepository                           │  │
│ │ - SQLiteTrainingRepository                          │  │
│ │ - SQLiteConfigRepository                            │  │
│ └──────────────────────────────────────────────────────┘  │
└────────────────────────┬──────────────────────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────────┐
         │ DISCO Y SQLITE                    │
         │ /data/bizsense.db (metadatos)    │
         │ /data/datasets/*.csv (datos)     │
         └───────────────────────────────────┘
```

### Estructura de Carpetas

```
BizSense/
├── app/                              # Frontend (Electron)
│   ├── main.js                       # Inicia backend + Electron
│   ├── preload.js                    # IPC bridge (seguridad)
│   ├── package.json
│   └── renderer/
│       ├── index.html                # Dashboard (página inicial)
│       ├── data.html                 # Cargar CSVs
│       ├── graphics.html             # Entrenar + visualizar
│       ├── models.html               # Descripción de modelos
│       ├── config.html               # Configuración global
│       ├── style.css                 # Tema oscuro
│       └── js/
│           ├── sidebar.js            # Navegación
│           ├── dashboard.js          # /stats endpoint
│           ├── data.js               # Upload + preview
│           ├── graphics.js           # Animación entrenamiento
│           ├── models.js             # (vacío actualmente)
│           └── config.js             # Guardar parámetros
│
├── ml/                               # Backend (Python)
│   ├── main.py                       # Inicia FastAPI
│   │
│   ├── domain/                       # Lógica pura (SIN dependencias externas)
│   │   ├── models.py                 # Dataset, Training, Config, Stats, IterationStep
│   │   └── services.py               # gradient_descent() con matemática
│   │
│   ├── application/                  # Orquestación (USE CASES)
│   │   ├── use_cases.py              # 9 casos de uso
│   │   └── ports/                    # Interfaces (HEXAGONAL)
│   │       ├── dataset_repository.py
│   │       ├── training_repository.py
│   │       └── config_repository.py
│   │
│   └── infrastructure/               # Adaptadores (FastAPI + SQLite)
│       ├── api/
│       │   ├── __init__.py
│       │   └── fastapi_adapter.py    # 11 endpoints HTTP
│       └── storage/
│           ├── __init__.py
│           └── sqlite_adapter.py     # Implementa 3 puertos
│
├── data/
│   ├── bizsense.db                   # Base de datos SQLite
│   ├── datasets/                     # Carpeta de CSVs cargados
│   │   └── (datasets subidos por usuario)
│   │
│   ├── pyme_sales.csv                # Dataset ejemplo (1 año de ventas)
│   ├── ventas_semanales.csv          # Agregación semanal
│   ├── clientes_churn.csv            # Para regresión logística (futuro)
│   ├── segmentacion_clientes.csv    # Para K-means (futuro)
│   ├── precio_optimo.csv             # Precio vs demanda
│   └── generate_datasets.py          # Script para regenerar
│
├── start.sh                          # Script de inicio unificado
├── requirements.txt                  # Dependencies Python
├── package.json                      # Dependencies Node
└── README.md                         # Este archivo
```

### Patrón de Arquitectura: Hexagonal

El proyecto sigue la **arquitectura hexagonal** (ports & adapters):

```
         ENTRADA (Driving)                SALIDA (Driven)
         FastAPI Adapter                  SQLite Adapter
              │                                │
              ▼                                ▼
         ┌──────────────────────────────────────────┐
         │         APPLICATION LAYER               │
         │     (USE CASES / ORCHESTRATION)        │
         │  UploadDataUseCase, TrainModelUseCase  │
         └──────────────────────────────────────────┘
                      │
                      ▼
         ┌──────────────────────────────────────────┐
         │          DOMAIN LAYER                    │
         │     (BUSINESS LOGIC - SIN SIDE EFFECTS)  │
         │  models.py: Dataset, Training, Config   │
         │  services.py: gradient_descent()        │
         └──────────────────────────────────────────┘
```

**Regla de dependencias:**
- ❌ domain NO depende de nada
- ❌ application solo depende de domain
- ✅ infrastructure implementa los puertos de application

---

## Flujo de Datos: De CSV a Predicción

### 1️⃣ Cargar Dataset (data.html)

```
Usuario selecciona CSV
         │
         ▼
fetch POST /datasets (file)
         │
         ▼
fastapi_adapter.py.upload_dataset()
         │
         ▼
UploadDataUseCase.execute()
         │
         ├─ Parsea CSV con pandas
         ├─ Crea Dataset(name, columns, rows, ...)
         ├─ Guarda en SQLite
         └─ Copia CSV a /data/datasets/{filename}
         │
         ▼
Retorna {id, name, columns, rows, preview}
         │
         ▼
Frontend lista datasets en tabla
```

### 2️⃣ Entrenar Modelo (graphics.html)

```
Usuario selecciona:
  - Dataset
  - Eje X (variable independiente)
  - Eje Y (variable a predecir)
  - Alpha (learning rate)
  - Iteraciones
         │
         ▼
fetch POST /train (JSON)
         │
         ▼
TrainModelUseCase.execute(dataset_id, x_col, y_col, alpha, iterations)
         │
         ├─ Obtiene Dataset completo
         ├─ Extrae columnas X e Y
         ├─ Llama gradient_descent() [DERIVADAS, MATEMÁTICA]
         │   └─ 200+ líneas de cálculo:
         │      - Normalización de X
         │      - Inicialización θ = [0, 0]
         │      - Para cada iteración:
         │        * h(x) = θ₀ + θ₁·x
         │        * Calcula error
         │        * Derivadas: ∂J/∂θ₀, ∂J/∂θ₁
         │        * Actualiza: θ := θ - α·∇J
         │        * Guarda history step
         │
         ├─ Retorna: {theta, history[100 steps], x_data, y_data, ...}
         │
         ├─ Guarda Training en SQLite
         └─ Guarda en training.history
         │
         ▼
Frontend recibe history
         │ 
         ├─ Renderiza scatter plot (datos + línea)
         ├─ Renderiza cost function evolution
         ├─ Puede reproducir paso a paso (con animación)
         └─ Muestra fórmulas en LaTeX (MathJax)
```

### 3️⃣ Predecir Nuevos Valores

```
Usuario ingresa x_values + usa theta entrenado
         │
         ▼
fetch POST /predict {x_values: [...], theta: [θ₀, θ₁]}
         │
         ▼
PredictUseCase.execute()
         │
         ├─ predictions = θ₀ + θ₁·x_values
         │
         ▼
Retorna {predictions: [y1, y2, ...]}
```

---

## Modelos de ML: Estado Actual

| # | Modelo | Estado | Cuándo | Concepto Matemático |
|---|--------|--------|--------|---------------------|
| 1 | **Regresión Lineal** | ✅ **IMPLEMENTADO** | Ahora | Derivadas + Gradiente |
| 2 | Regresión Logística | ⏳ Pendiente | v1.1 | Sigmoide + Derivada |
| 3 | K-Means | ⏳ Pendiente | v1.2 | Límites (convergencia) |
| 4 | Red Neuronal | ⏳ Pendiente | v2.0 | Backprop (regla cadena) |

---

## Endpoints de la API

Todos retornan **JSON**. Host: `http://localhost:5000`

### Estadísticas & Dashboard

| Endpoint | Método | Descripción | Respuesta |
|----------|--------|-------------|----------|
| `/stats` | GET | Métricas para dashboard | `{datasets_count, trainings_count, models_implemented, last_training}` |

### Datasets

| Endpoint | Método | Auth | Descripción |
|----------|--------|------|-------------|
| `/datasets` | GET | - | Lista todos (sin datos) |
| `/datasets` | POST | - | Subir CSV. Body: `multipart/form-data` con `file` |
| `/datasets/{id}` | GET | - | Preview (primeras 10 filas) |
| `/datasets/{id}` | DELETE | - | Eliminar dataset + CSV |

**Ejemplo CREATE:**
```bash
curl -X POST http://localhost:5000/datasets \
  -F "file=@ventas.csv"
```

**Respuesta:**
```json
{
  "id": 1,
  "name": "ventas.csv",
  "columns": ["fecha", "cantidad", "precio"],
  "rows": 156,
  "preview": [
    {"fecha": "2024-01-01", "cantidad": 150, "precio": 25000},
    ...
  ],
  "created_at": "2024-06-26T10:30:00"
}
```

### Entrenamientos

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/train` | POST | Entrenar modelo. Body: `{dataset_id, x_col, y_col, alpha, iterations}` |
| `/trainings` | GET | Historial de entrenamientos (sin history) |
| `/trainings/{id}` | GET | Detalle de 1 entrenamiento (sin history) |

**Ejemplo TRAIN:**
```bash
curl -X POST http://localhost:5000/train \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "x_col": "semana",
    "y_col": "ventas",
    "alpha": 0.01,
    "iterations": 100
  }'
```

**Respuesta:**
```json
{
  "theta": [1234.5, 56.78],
  "history": [
    {
      "iteration": 1,
      "theta_0": 0.0,
      "theta_1": 0.0,
      "theta_0_real": 1200.0,
      "theta_1_real": 50.0,
      "cost": 1234567.89,
      "gradient_0": -12.34,
      "gradient_1": -5.67,
      "predictions": [1250.0, 1306.78, ...]
    },
    ...
  ],
  "x_data": [1, 2, 3, ...],
  "y_data": [1250, 1306, ...],
  "x_norm": {"mean": 78.5, "std": 45.2}
}
```

### Configuración

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/config` | GET | Obtener configuración actual |
| `/config` | POST | Actualizar config. Body: `{default_alpha, default_iterations, theme, language}` |

### Predicción

| Endpoint | Método | Descripción |
|----------|--------|-------------|
| `/predict` | POST | Predecir con theta. Body: `{x_values: [...], theta: [θ₀, θ₁]}` |

---

## Casos de Uso (Use Cases)

| Caso | Entrada | Salida | Responsabilidad |
|------|---------|--------|-----------------|
| `UploadDataUseCase` | filename, file_content | Dataset (id, metadata) | Parsear CSV → guardar BD + disco |
| `ListDatasetsUseCase` | - | List[Dataset] | Listar metadatos (sin data) |
| `GetDatasetUseCase` | dataset_id, load_data | Dataset | Obtener con datos opcionales |
| `DeleteDatasetUseCase` | dataset_id | - | Eliminar de BD + disco |
| `TrainModelUseCase` | dataset_id, x_col, y_col, α, iter | TrainingResult + historia | Ejecutar gradiente descendente |
| `GetTrainingHistoryUseCase` | - | List[Training] | Listar entrenamientos históricos |
| `GetStatsUseCase` | - | Stats | Agregados para dashboard |
| `ManageConfigUseCase` | - | AppConfig | Leer/actualizar parámetros globales |
| `PredictUseCase` | x_values, theta | List[float] | Aplicar ecuación lineal |

---

## Páginas de la UI

| Página | Ruta | Descripción |
|--------|------|-------------|
| **Dashboard** | `/app/renderer/index.html` | **INICIO**: Métricas, último entrenamiento, acceso rápido |
| **Datos** | `/app/renderer/data.html` | Subir CSV, listar, previsualizar, eliminar |
| **Gráfico** | `/app/renderer/graphics.html` | **PRINCIPAL**: Entrenar + visualizar paso a paso |
| **Modelos** | `/app/renderer/models.html` | Descripción teórica de cada modelo (solo lectura) |
| **Config** | `/app/renderer/config.html` | Alpha default, iteraciones, tema, idioma |

---

## Datasets de Ejemplo

En `/data/` hay 6 archivos CSV generados por `generate_datasets.py`:

| Archivo | Rows | Columnas | Uso | Contexto |
|---------|------|----------|-----|----------|
| **pyme_sales.csv** | 156 | semana, ventas, margen, region | ✅ Regresión lineal | Ventas semanales por región |
| **ventas_semanales.csv** | 156 | semana, total_vendido, costo | ✅ Regresión lineal | Agregado semanal |
| **clientes_churn.csv** | 5000 | cliente_id, antiguedad, compras, estado | ⏳ Regresión logística | Predicción: se va o se queda |
| **segmentacion_clientes.csv** | 5000 | cliente_id, frecuencia, monto_gasto, region | ⏳ K-means | Agrupar por tipo cliente |
| **precio_optimo.csv** | 50 | precio, demanda, ganancia | ✅ Regresión lineal | Curva precio-demanda |
| **sales_sample.csv** | Pequeño | date, amount | ✅ Para pruebas rápidas |  |

**Generarlos nuevamente:**
```bash
python data/generate_datasets.py
```

---

## Desarrollo Local

### Estructura para Contribuciones

```
ml/domain/
  ├─ services.py        # Agregar nuevo algoritmo aquí
  └─ models.py          # Nuevas entidades

ml/application/
  ├─ use_cases.py       # Nuevo caso de uso
  └─ ports/
     └─ nuevo_repository.py  # Nueva interfaz

ml/infrastructure/
  ├─ api/
  │  └─ fastapi_adapter.py   # Nuevos @app.post()/@app.get()
  └─ storage/
     └─ sqlite_adapter.py    # Implementar repositorio

app/renderer/
  ├─ js/
  │  └─ nuevo.js       # Nueva lógica UI
  └─ nuevo.html       # Nueva página
```

### Debugging

**Backend:**
```bash
# Ver logs de FastAPI
tail -f /tmp/bizsense-backend.log

# O correr directamente (con stack traces)
python -m ml.main
```

**Frontend:**
```bash
# Ver DevTools: Ctrl+Shift+I en Electron
npm start -- --dev
```

**Base de datos:**
```bash
# Inspeccionar SQLite
sqlite3 data/bizsense.db

sqlite> SELECT * FROM datasets;
sqlite> SELECT * FROM trainings;
sqlite> SELECT * FROM config;
```

---

## Troubleshooting

### ❌ "Backend no respondió después de 10 segundos"

**Causa:** FastAPI tarda en iniciar o error en imports

**Solución:**
```bash
# Ejecutar backend directamente
source .venv/bin/activate
python -m ml.main

# Debería ver: "Uvicorn running on http://0.0.0.0:5000"
```

### ❌ "ModuleNotFoundError: No module named 'fastapi'"

**Solución:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### ❌ "CORS error when fetching from frontend"

**Causa:** Backend no está corriendo o puerto incorrecto

**Verificar:**
```bash
curl http://localhost:5000/stats
# Debería retornar JSON, no rechazo de conexión
```

### ❌ "Electron no abre / ventana negra"

**Solución:**
```bash
# Instalar dependencias Node
npm install

# Reintentar
npm start -- --dev
```

### ❌ "AttributeError en gradient_descent"

**Causa Común:** Columna con tipos no numéricos (strings, NaN)

**Solución:** Verificar CSV tiene solo números en X e Y
```bash
# En el CSV, remover header o añadir filtro
```

---

## Licencia

MIT

---

## Próximos Pasos (Roadmap)

- [ ] **v1.1:** Regresión Logística (binaria)
- [ ] **v1.2:** K-Means (clustering)
- [ ] **v2.0:** Red Neuronal simple (backprop)
- [ ] **v2.1:** Validación cruzada + test/train split
- [ ] **v3.0:** Exportar modelo a JSON/joblib
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