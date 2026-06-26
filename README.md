![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Node.js](https://img.shields.io/badge/Node.js-18+-green?logo=node.js)
![Electron](https://img.shields.io/badge/Electron-Desktop-9cf?logo=electron)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-yellow)

# 📊 BizSense
## Interactive Machine Learning with Calculus

**Aplicación interactiva que visualiza cómo el Cálculo Diferencial (derivadas, gradientes) permite que las máquinas aprendan.**

> Proyecto académico para **Cálculo Diferencial**  
> Autores: Benjamin Cifuentes, Benjamin Contesso

---

## 🚀 Quick Start

```bash
./start.sh
```

**Backend:** `http://localhost:5000`  
**Frontend:** Electron app

---

## 📌 Key Features

| Característica | Qué hace |
|---|---|
| 📤 **Upload CSV** | Sube datos, visualiza preview |
| 🧠 **Entrenar modelo** | Calcula derivadas en tiempo real (step-by-step) |
| 📈 **Animación** | Ve cómo el algoritmo mejora iteración tras iteración |
| 🎯 **Predicciones** | Usa la ecuación aprendida para predecir nuevos valores |
| 💾 **Persistencia** | Guarda todo en SQLite |

---

## 🎓 Conceptos de Cálculo Diferencial Usados

```
Regresión Lineal = Minimizar función de costo mediante gradiente descendente

h(x) = θ₀ + θ₁·x                      ← Hipótesis (línea)
J(θ) = (1/2m) Σ(h(x_i) - y_i)²        ← Función de costo (error)
∂J/∂θ = (1/m) Σ(h(x_i) - y_i)·x_i     ← Derivada (gradiente)
θ := θ - α·∇J                          ← Actualización (descenso)
```

**Concepto central:** Las **derivadas** nos dicen hacia dónde movernos para minimizar el error.

---

## 🏗️ Architecture

```
┌─────────────────┐
│    ELECTRON     │  UI + Gráficos (Chart.js, MathJax)
├─────────────────┤
│    FastAPI      │  11 endpoints REST
├─────────────────┤
│ Use Cases + ML  │  Lógica de negocio (gradiente descendente)
├─────────────────┤
│  SQLite + CSV   │  Persistencia
└─────────────────┘
```

**Patrón:** Arquitectura Hexagonal (puertos + adaptadores)

---

## 📁 Project Structure

```
BizSense/
├── app/                    # Frontend (Electron)
│   ├── main.js            # Inicia app
│   └── renderer/           # Páginas HTML + JS
│
├── ml/                     # Backend (Python)
│   ├── domain/            # Lógica pura (gradient_descent)
│   ├── application/       # 9 casos de uso
│   └── infrastructure/    # FastAPI + SQLite
│
├── data/                  # Base de datos + CSVs
│   ├── bizsense.db       # Metadatos
│   ├── datasets/         # CSVs subidos
│   └── *.csv            # Ejemplos
│
└── start.sh              # Script arranque
```

---

## 📊 How It Works

| Paso | Qué pasa | Math |
|------|----------|------|
| 1️⃣ Subir CSV | User carga datos | `Dataset(name, columns, rows)` |
| 2️⃣ Entrenar | Calcula parámetros | `θ := θ - α·∇J` (50-100 iteraciones) |
| 3️⃣ Ver progreso | Animación step-by-step | Muestra cost function decreciendo |
| 4️⃣ Predecir | Usa ecuación final | `y = θ₀ + θ₁·x` |

---

## 🔌 API Endpoints

| Endpoint | Método | Descripción |
|----------|--------|---|
| `/stats` | GET | Métricas dashboard |
| `/datasets` | GET/POST | Listar / subir |
| `/train` | POST | Entrenar modelo |
| `/predict` | POST | Predecir valores |
| `/config` | GET/POST | Guardar parámetros |

---

## 📋 Modelos Implementados

| # | Modelo | Estado | Concepto Cálculo |
|---|--------|--------|---|
| 1 | **Regresión Lineal** | ✅ | Derivadas + Gradientes |
| 2 | Regresión Logística | ⏳ | Función sigmoide |
| 3 | K-Means | ⏳ | Límites (convergencia) |
| 4 | Red Neuronal | ⏳ | Backprop (regla cadena) |

---

## 📚 Datasets de Ejemplo

```
data/
├── pyme_sales.csv           ← Sale aquí (para regresión lineal)
├── ventas_semanales.csv
├── clientes_churn.csv       ← Para regresión logística (futuro)
├── segmentacion_clientes.csv ← Para K-means (futuro)
└── precio_optimo.csv
```

---

## 🛠️ Development

```bash
# Backend logs
tail -f /tmp/bizsense-backend.log

# Frontend DevTools
npm start -- --dev

# Inspeccionar BD
sqlite3 data/bizsense.db
SELECT * FROM datasets;
SELECT * FROM trainings;
```

---

## ❌ Troubleshooting

| Problema | Solución |
|----------|----------|
| Backend no responde | `python -m ml.main` (verificar errores) |
| Falta módulo `fastapi` | `pip install -r requirements.txt` |
| CORS error | Verificar: `curl http://localhost:5000/stats` |
| Electron no abre | `npm install && npm start` |

---

## 📖 Techs Used

- **Frontend:** Electron, Chart.js, MathJax
- **Backend:** FastAPI, Python 3.8+
- **ML:** NumPy, Pandas
- **Database:** SQLite
- **Pattern:** Hexagonal Architecture

---

## 📜 License

MIT

---

## 🚀 Next Steps (Roadmap)

- [ ] v1.1: Regresión Logística
- [ ] v1.2: K-Means  
- [ ] v2.0: Red Neuronal
- [ ] v3.0: Exportar modelos