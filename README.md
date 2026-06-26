# BizSense - Sistema de Analisis Predictivo para Pymes

## Descripcion del Proyecto

BizSense es una aplicacion de escritorio desarrollada para ayudar a pequenas y medianas empresas (pymes) a tomar decisiones basadas en datos. La aplicacion permite predecir ventas, analizar la retencion de clientes y optimizar precios mediante modelos de Machine Learning.

## Integrantes del Equipo

- Benjamin Cifuentes
- Benjamin Contesso

## Asignatura

Calculo Diferencial

## Docente

Marcelo Andres Sepulveda Albornoz

## Instalación y Ejecución

### Requisitos previos
- Python 3.8+
- Node.js 18+
- npm

### Ejecutar la aplicación

```bash
./start.sh
```

Este comando:
1. Crea y activa el entorno virtual de Python
2. Instala todas las dependencias
3. Inicia el backend (FastAPI) en puerto 5000
4. Inicia la aplicación Electron

## Arquitectura

El proyecto sigue una **arquitectura hexagonal** (puertos y adaptadores):

```
ml/
├── domain/              # Lógica de negocio pura
│   ├── models.py        # Entidades (Dataset, TrainingResult, IterationStep)
│   └── services.py      # Servicios de dominio (gradiente descendente)
├── application/         # Casos de uso
│   └── use_cases.py     # UploadDataUseCase, TrainModelUseCase
├── infrastructure/      # Adaptadores
│   └── api/
│       └── fastapi_adapter.py  # API HTTP
└── main.py              # Punto de entrada

app/
├── main.js              # Electron main process
├── preload.js           # Bridge seguro
└── renderer/
    ├── index.html       # Dashboard UI
    ├── renderer.js      # Lógica frontend + Chart.js + MathJax
    └── style.css        # Estilos
```

## Modelos de Machine Learning

### 1. Regresion Lineal (Implementado)
- Uso: Prediccion de ventas futuras
- Derivadas: Descenso por gradiente para minimizar el error
- Visualización: Animación paso a paso del entrenamiento

### 2. Regresion Logistica (Próximamente)
- Uso: Clasificacion (ventas buenas/malas, clientes que se van/que se quedan)
- Derivadas: Funcion sigmoide y su derivada

### 3. K-Means (Próximamente)
- Uso: Clustering (segmentar negocios o productos)
- Limites: Convergencia cuando n tiende a infinito

### 4. Red Neuronal (Próximamente)
- Uso: Detectar patrones complejos
- Derivadas: Backpropagation (regla de la cadena)

## Calculo Diferencial en el Proyecto

### Derivadas
- Descenso por gradiente: minimos de la funcion de costo
- Optimizacion: encontrar maximos y minimos de funciones de ganancia/perdida
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
| Arquitectura | Hexagonal (Puertos y Adaptadores) |

## Licencia

MIT
