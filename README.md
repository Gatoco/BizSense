```
 ___ _        _____
| _ (_)___   |__ / ___ _ __  ___ _ _  __ _ _ __  ___
| _ \ / -_)   |_ \/ _ \ '_ \/ -_) ' \/ _` | '  \/ -_)
|___/_\___|  |___/\___/ .__/\___|_||_\__,_|_|_|_\___|
                      |_|
       Sistema de Analisis Predictivo para Pymes
```

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Electron](https://img.shields.io/badge/Electron-Desktop-9cf?logo=electron)](https://electronjs.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-FF6384?logo=chartdotjs)](https://chartjs.org)
[![MathJax](https://img.shields.io/badge/MathJax-LaTeX-008080)](https://mathjax.org)
[![SQLite](https://img.shields.io/badge/SQLite-003B57?logo=sqlite)](https://sqlite.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy)](https://numpy.org)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas)](https://pandas.pydata.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_AI-000?logo=ollama)](https://ollama.com)
[![LM Studio](https://img.shields.io/badge/LM_Studio-Local_AI-00d4ff)](https://lmstudio.ai)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Descripcion

BizSense es una aplicacion de escritorio que ayuda a pequenas y medianas empresas a tomar decisiones basadas en datos. Visualiza como el Calculo Diferencial (derivadas, gradientes, limites) permite que las maquinas aprendan. Disenada para ser intuitiva para trabajadores de pymes con resultados en lenguaje simple, con asistencia de IA local para responder dudas.

Proyecto academico para **Calculo Diferencial**  
Autores: Benjamin Cifuentes, Benjamin Contesso

---

## Quick Start

```bash
./start.sh
```

**Backend:** `http://localhost:5000`  
**Frontend:** Ventana Electron independiente  
**IA Local:** Detecta automaticamente LM Studio (puerto 1234) u Ollama (puerto 11434)

---

## Key Features

| Caracteristica | Que hace |
|---|---|
| **Upload CSV** | Sube datos, visualiza preview en tabla |
| **4 Modelos ML** | Regresion Lineal, Logistica, K-Means, Red Neuronal |
| **Animacion paso a paso** | Control Play/Pausa/Paso + slider de velocidad |
| **Formulas en tiempo real** | LaTeX con MathJax, valores numericos en cada iteracion |
| **Resultados en lenguaje simple** | Metricas no tecnicas: tendencia, prediccion, confianza |
| **IA Local** | Chat con modelo local (Ollama/LM Studio) para responder dudas |
| **Persistencia** | SQLite guarda datasets, entrenamientos, historial de chat |
| **Tema oscuro** | Style LM Studio (bg #0f0f23, cyan #00d4ff) |
| **Dashboard** | Metricas globales, ultimo entrenamiento, acceso rapido |

---

## Conceptos de Calculo Diferencial Usados

### Derivadas (Gradiente Descendente)
```
Regresion Lineal: Minimizar funcion de costo

h(x) = theta_0 + theta_1 * x                           -> Hipotesis (linea)
J(theta) = (1/2m) * sum(h(x_i) - y_i)^2                -> Funcion de costo (error)
dJ/dtheta_0 = (1/m) * sum(h(x_i) - y_i)                 -> Derivada parcial 1
dJ/dtheta_1 = (1/m) * sum(h(x_i) - y_i) * x_i           -> Derivada parcial 2
theta_j := theta_j - alpha * dJ/dtheta_j                -> Actualizacion (descenso)
```

### Funcion Sigmoide (Regresion Logistica)
```
g(z) = 1 / (1 + e^-z)
Limites: g(z) -> 1 cuando z -> +inf
         g(z) -> 0 cuando z -> -inf
```

### Limites (Convergencia en K-Means)
```
Cuando las iteraciones tienden a infinito, las asignaciones
de clusters y los centroides dejan de cambiar.
```

### Regla de la Cadena (Backpropagation en Red Neuronal)
```
delta^(L) = a^(L) - y                           -> Error en capa de salida
delta^(l) = (W^(l+1))^T * delta^(l+1) .* g'(z) -> Propagacion hacia atras
dJ/dW = delta * a^T                              -> Gradiente de pesos
```

---

## Arquitectura (Hexagonal - Puertos y Adaptadores)

```
app/ (Electron)
  main.js          -> Spawnea backend Python, espera a que este listo
  renderer/
    index.html     -> Dashboard (metricas, ultimo entrenamiento)
    models.html    -> Lista de modelos con formulas
    graphics.html  -> Entrenamiento animado paso a paso
    results.html   -> Resultados no tecnicos + chat con IA
    data.html      -> Gestion de datasets CSV
    config.html    -> Configuracion global + IA
    style.css      -> Tema oscuro LM Studio
    js/
      sidebar.js      -> Navegacion inyectada dinamicamente
      dashboard.js    -> Logica dashboard
      graphics.js     -> Animacion por modelo (4 tipos)
      results.js      -> Metricas + grafico prediccion
      chat.js         -> Chat streaming con IA local
      data.js         -> CRUD datasets
      config.js       -> Guardar/cargar config

ml/ (Python - FastAPI)
  domain/                   -> Capa interna (logica pura)
    models.py               -> Entidades: Dataset, Training, Config, Stats, etc.
    services.py             -> 4 servicios ML: gradient_descent, logistic, kmeans, neural

  application/              -> Capa media (casos de uso)
    use_cases.py            -> 13 casos de uso
    ports/                  -> Puertos (interfaces abstractas)
      dataset_repository.py -> save, get, list, delete
      training_repository.py-> save, get, list, get_last
      config_repository.py  -> get_all, update
      ai_provider.py        -> is_available, chat, list_models, start, pull

  infrastructure/           -> Capa externa (adaptadores)
    api/
      fastapi_adapter.py    -> 17 endpoints REST
    storage/
      sqlite_adapter.py     -> Implementa los puertos con SQLite
    ai/
      ollama_adapter.py     -> Provider local via Ollama (puerto 11434)
      lmstudio_adapter.py   -> Provider local via LM Studio (puerto 1234)

data/
  bizsense.db               -> SQLite (datasets, trainings, config, chat_history)
  datasets/                 -> CSVs subidos por el usuario
  *.csv                     -> Datasets de ejemplo
```

**Flujo de dependencias:**
```
infrastructure/ -> application/ -> domain/
  (adaptadores)   (casos de uso)  (logica pura)
```

---

## Modelos Implementados

| # | Modelo | Estado | Concepto de Calculo | Visualizacion |
|---|--------|--------|---------------------|---------------|
| 1 | Regresion Lineal | Implementado | Derivadas (gradiente descendente) | Linea recta que se ajusta |
| 2 | Regresion Logistica | Implementado | Sigmoide + limites | Curva sigmoide |
| 3 | K-Means | Implementado | Limites (convergencia) | Puntos coloreados + centroides |
| 4 | Red Neuronal | Implementado | Regla de la cadena (backprop) | Curva de prediccion |

---

## API Endpoints

| Endpoint | Metodo | Descripcion |
|----------|--------|-------------|
| `/stats` | GET | Metricas del dashboard |
| `/datasets` | GET | Listar datasets |
| `/datasets` | POST | Subir CSV |
| `/datasets/{id}` | GET | Preview de dataset |
| `/datasets/{id}` | DELETE | Eliminar dataset |
| `/train` | POST | Entrenar modelo (4 tipos) |
| `/trainings` | GET | Historial de entrenamientos |
| `/trainings/{id}` | GET | Detalle de entrenamiento |
| `/predict` | POST | Predecir con modelo entrenado |
| `/insights` | GET | Metricas no tecnicas del ultimo entrenamiento |
| `/config` | GET | Configuracion actual |
| `/config` | POST | Actualizar configuracion |
| `/ai/status` | GET | Estado del proveedor IA |
| `/ai/models` | GET | Modelos disponibles |
| `/ai/start` | POST | Iniciar servidor local |
| `/ai/chat` | POST | Chat streaming con IA (SSE) |
| `/chat/history` | GET | Historial de chat |
| `/chat/history` | DELETE | Limpiar historial |

---

## Datasets de Ejemplo

```
data/
  pyme_sales.csv              -> Para regresion lineal (ventas por tiempo)
  ventas_semanales.csv        -> Para regresion lineal
  clientes_churn.csv          -> Para regresion logistica (churn binario)
  segmentacion_clientes.csv   -> Para K-Means (clusters)
  precio_optimo.csv           -> Para optimizacion de precios
  sales_sample.csv            -> Datos de ventas simple (52 semanas)
  generate_datasets.py        -> Generador de datasets sinteticos
```

---

## Como Usar

1. **Subir datos**: Ir a Datos -> Seleccionar CSV -> se guarda automaticamente
2. **Entrenar modelo**: Ir a Funcion Grafica
   - Seleccionar dataset, modelo, columnas X/Y
   - Configurar alpha e iteraciones
   - Click "Entrenar modelo"
   - Usar Play/Pausa/Paso + slider de velocidad
   - Ver formulas LaTeX con valores en tiempo real
3. **Ver resultados**: Ir a Resultados
   - Metricas en lenguaje simple: tendencia, prediccion, confianza
   - Grafico de prediccion futura (linea punteada)
   - Chat con IA local para preguntas sobre los resultados
4. **Configurar**: Ir a Configuracion
   - Parametros de entrenamiento, apariencia, proveedor IA

---

## Asistente IA Local

BizSense soporta dos proveedores de IA local:

| Proveedor | Puerto | Deteccion | Modelos recomendados |
|-----------|--------|-----------|---------------------|
| **Ollama** | 11434 | Automatica | qwen2.5:0.5b, qwen2.5:1.5b, tinyllama:1.1b |
| **LM Studio** | 1234 | Automatica | qwen2.5-coder-14b, gemma-4-12b, phi-3 |

- Se detecta automaticamente al abrir la pagina Resultados
- Usa el modelo activo de LM Studio o el configurado en Ollama
- El chat es streaming (token a token)
- El historial es persistente en SQLite
- System prompt con contexto de los resultados reales del modelo entrenado

---

## Development

```bash
# Backend logs
tail -f /tmp/bizsense-backend.log

# Electron devtools
npm start -- --dev

# Inspeccionar BD
sqlite3 data/bizsense.db
SELECT * FROM datasets;
SELECT * FROM trainings;
```

---

## Troubleshooting

| Problema | Solucion |
|----------|----------|
| Backend no responde | `python -m ml.main` (verificar errores) |
| Falta modulo Python | `pip install -r requirements.txt` |
| Electron no abre | `npm install && npm start` |
| IA no disponible | Abrir LM Studio con un modelo cargado, o iniciar `ollama serve` |
| Puerto 5000 ocupado | `fuser -k 5000/tcp` |

---

## Tech Stack

- **Frontend:** Electron, HTML/CSS/JS Vanilla, Chart.js, MathJax
- **Backend:** FastAPI, Python 3.8+, Uvicorn
- **ML:** NumPy, Pandas (calculos puros, sin scikit-learn)
- **IA Local:** Ollama, LM Studio (auto-deteccion)
- **Database:** SQLite (stdlib)
- **Pattern:** Arquitectura Hexagonal (Puertos y Adaptadores)
- **Auth:** Ninguna (aplicacion local de escritorio)

---

## Licencia

MIT