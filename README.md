```
 ____  _              _
| __ )| | _____  _   _| |_ ___ _ __ ___
|  _ \| |/ _ \ \/ / / __/ _ \ '__/ _ \
| |_) | | (_) >  < | ||  __/ | |  __/
|____/|_|\___/_/\_\ \__\___|_|  \___|
        Analisis Predictivo para Pymes
```

[![Calculo Diferencial](https://img.shields.io/badge/Calculo-Diferencial-9cf)]()
[![Arquitectura Hexagonal](https://img.shields.io/badge/Arquitectura-Hexagonal-00d4ff)]()
[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Electron](https://img.shields.io/badge/Electron-Desktop-9cf?logo=electron)](https://electronjs.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite)](https://sqlite.org)
[![NumPy](https://img.shields.io/badge/NumPy-ML-013243?logo=numpy)](https://numpy.org)
[![IA Local](https://img.shields.io/badge/IA-Local_Ollama_LM_Studio-00d4ff)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

---

## Que es BizSense?

BizSense es una aplicacion de escritorio que permite a pequenas y medianas empresas **predecir ventas, segmentar clientes y optimizar precios** mediante modelos de Machine Learning.

La particularidad de BizSense es que **visualiza el calculo matematico detras del aprendizaje automatico**: el usuario puede observar, paso a paso, como las **derivadas** guian al modelo hacia la solucion, como los **limites** explican la convergencia, y como la **regla de la cadena** permite el backpropagation en redes neuronales.

> Proyecto academico para **Calculo Diferencial**
> Autores: Benjamin Cifuentes, Benjamin Contesso
> Docente: Marcelo Andres Sepulveda Albornoz

---

## Key Features

- **4 modelos de Machine Learning** implementados desde cero (sin librerias de ML)
- **Visualizacion paso a paso** del entrenamiento con control de velocidad
- **Formulas matematicas en tiempo real** renderizadas con LaTeX
- **Resultados en lenguaje simple** para trabajadores de pymes (no tecnicos)
- **Asistente de IA local** que responde dudas sobre los resultados
- **Persistencia completa** con SQLite (datasets, entrenamientos, historial de chat)
- **Tema oscuro** inspirado en LM Studio

---

## Modelos y Conceptos de Calculo

### 1. Regresion Lineal -- Derivadas (Gradiente Descendente)

Predice valores continuos (ej: ventas futuras) ajustando una linea recta.

```
h(x) = theta_0 + theta_1 * x                        Hipotesis
J(theta) = (1/2m) * sum(h(x_i) - y_i)^2             Funcion de costo
dJ/dtheta_j = (1/m) * sum(h(x_i) - y_i) * x_i        Derivada parcial
theta_j := theta_j - alpha * dJ/dtheta_j             Actualizacion
```

**Concepto clave:** Las derivadas parciales indican la direccion de maximo crecimiento del error. El gradiente descendente se mueve en direccion contraria para minimizarlo.

---

### 2. Regresion Logistica -- Derivadas de la Sigmoide + Limites

Clasifica en dos categorias (ej: cliente se queda / se va).

```
g(z) = 1 / (1 + e^(-z))                              Funcion sigmoide
h(x) = g(theta_0 + theta_1 * x)                      Probabilidad
J(theta) = -(1/m) * sum[y*log(h) + (1-y)*log(1-h)]   Cross-entropy
```

**Concepto clave (limites):**
- Cuando `z -> +inf`, `g(z) -> 1` (clasificacion positiva segura)
- Cuando `z -> -inf`, `g(z) -> 0` (clasificacion negativa segura)

---

### 3. K-Means -- Limites (Convergencia)

Agruta datos en `k` clusters por similitud.

```
c(i) = argmin_j ||x(i) - mu_j||^2                    Asignacion al cluster mas cercano
mu_j = (1/|C_j|) * sum_{i in C_j} x(i)               Actualizacion del centroide
J = sum ||x(i) - mu_c(i)||^2                          Funcion de costo (inercia)
```

**Concepto clave (limites):** Cuando el numero de iteraciones `n -> inf`, las asignaciones y centroides dejan de cambiar. El algoritmo **converge** a un punto fijo.

---

### 4. Red Neuronal -- Regla de la Cadena (Backpropagation)

Aprende patrones complejos mediante una capa oculta.

```
Forward:
  z(1) = W(1) * x + b(1)         a(1) = g(z(1))      Capa oculta
  z(2) = W(2) * a(1) + b(2)      a(2) = g(z(2))      Salida

Backpropagation (regla de la cadena):
  delta(2) = a(2) - y                                  Error en salida
  delta(1) = (W(2))^T * delta(2) .* g'(z(1))          Propagacion hacia atras
  dJ/dW(2) = delta(2) * (a(1))^T                       Gradiente capa 2
  dJ/dW(1) = delta(1) * x^T                             Gradiente capa 1
```

**Concepto clave:** La regla de la cadena permite descomponer la derivada del error total en las derivadas de cada capa, propagando el error desde la salida hacia la entrada.

---

## Arquitectura Hexagonal

BizSense sigue el patron de **Puertos y Adaptadores** (arquitectura hexagonal), que separa la logica de negocio de los detalles tecnicos.

```
                    CAPA INTERNA
                 (logica pura, sin dependencias)
                +-----------------------+
                |       Domain          |
                |  - Modelos ML (4)     |
                |  - Entidades          |
                |  - Calculo puro       |
                +-----------+-----------+
                            |
                +-----------+-----------+
                |     Application       |
                |  - 13 casos de uso    |
                |  - Puertos (interfaces)|
                +-----------+-----------+
                            |
            +---------------+---------------+
            |                               |
   CAPA EXTERNA                      CAPA EXTERNA
  (adaptadores entrada)           (adaptadores salida)
  +----------------+              +----------------+
  |   FastAPI      |              |   SQLite       |
  |   17 endpoints |              |   Persistencia |
  +----------------+              +----------------+
  +----------------+              +----------------+
  |   Electron     |              |   Ollama       |
  |   UI Desktop   |              |   LM Studio    |
  +----------------+              +----------------+
```

**Principio:** Las dependencias apuntan hacia adentro. El dominio (calculo matematico) no sabe nada de FastAPI, SQLite, ni Electron. Los adaptadores implementan las interfaces que el dominio define.

### Puertos (interfaces abstractas)

| Puerto | Operaciones |
|--------|-------------|
| `DatasetRepository` | guardar, obtener, listar, eliminar datasets |
| `TrainingRepository` | guardar, obtener, listar, ultimo entrenamiento |
| `ConfigRepository` | leer, actualizar configuracion |
| `AIProvider` | verificar disponibilidad, chat, listar modelos, iniciar |

### Adaptadores (implementaciones)

| Adaptador | Implementa | Tecnologia |
|-----------|-----------|------------|
| `FastAPIAdapter` | Endpoints HTTP | FastAPI |
| `SQLiteAdapter` | Repositorios | sqlite3 (stdlib) |
| `OllamaAdapter` | AIProvider | Ollama API (puerto 11434) |
| `LMStudioAdapter` | AIProvider | LM Studio API (puerto 1234) |

---

## Flujo de Uso

```
1. Datos         Subir CSV (se guarda en SQLite + disco)
      |
2. Funcion       Seleccionar dataset + modelo + parametros
   Grafica       Click "Entrenar"
      |          Ver animacion paso a paso:
      |          - Grafico del modelo ajustandose
      |          - Funcion de costo decreciendo
      |          - Formulas con valores en cada iteracion
      |          Controles: Play / Pausa / Paso adelante / Paso atras
      |
3. Resultados    Metricas en lenguaje simple:
      |          - "Tus ventas suben ~$200 por semana"
      |          - "Prediccion proxima semana: $25,400"
      |          - "Confianza del modelo: alta"
      |          Grafico de prediccion futura
      |          Chat con IA local para preguntas
      |
4. Dashboard     Resumen global: datasets, entrenamientos, acceso rapido
```

---

## Estructura del Proyecto

```
BizSense/
├── app/                         Frontend (Electron)
│   ├── main.js                  Inicia backend + ventana
│   └── renderer/
│       ├── index.html           Dashboard
│       ├── models.html          Catalogo de modelos
│       ├── graphics.html        Entrenamiento animado
│       ├── results.html         Resultados + chat IA
│       ├── data.html            Gestion de datasets
│       ├── config.html          Configuracion
│       ├── style.css            Tema oscuro
│       └── js/                  Logica por pagina
│
├── ml/                          Backend (Python)
│   ├── domain/                  Capa interna
│   │   ├── models.py            Entidades
│   │   └── services.py          4 algoritmos ML
│   ├── application/             Capa media
│   │   ├── use_cases.py         13 casos de uso
│   │   └── ports/               4 interfaces
│   └── infrastructure/          Capa externa
│       ├── api/                 FastAPI (17 endpoints)
│       ├── storage/             SQLite
│       └── ai/                  Ollama + LM Studio
│
├── data/                        Datos
│   ├── bizsense.db              SQLite
│   ├── datasets/                CSVs subidos
│   └── *.csv                    Datasets de ejemplo
│
└── start.sh                     Script de inicio
```

---

## Tech Stack

| Capa | Tecnologia |
|------|-----------|
| Frontend | Electron + HTML/CSS/JS + Chart.js + MathJax |
| Backend | FastAPI + Uvicorn |
| ML | NumPy + Pandas (algoritmos propios, sin scikit-learn) |
| IA | Ollama + LM Studio (auto-deteccion) |
| Storage | SQLite (stdlib) |
| Arquitectura | Hexagonal (Puertos y Adaptadores) |

---

## Ejecucion

```bash
./start.sh
```

Requisitos: Python 3.8+, Node.js 18+

---

## Licencia

MIT