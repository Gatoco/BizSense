const API = 'http://localhost:5000';
let predictionChart = null;

document.addEventListener('DOMContentLoaded', async () => {
    await loadInsights();
});

async function loadInsights() {
    try {
        const res = await fetch(`${API}/insights`);
        const data = await res.json();

        if (!data.has_data) {
            document.getElementById('no-data').classList.remove('hidden');
            document.getElementById('results-content').classList.add('hidden');
            return;
        }

        document.getElementById('no-data').classList.add('hidden');
        document.getElementById('results-content').classList.remove('hidden');

        updateModelInfo(data);
        updateTrendMetrics(data);
        updateInterpretation(data);
        renderChart(data);
    } catch (err) {
        console.error('Error loading insights:', err);
    }
}

function updateModelInfo(data) {
    document.getElementById('model-type').textContent = data.model_type.replace('_', ' ').toUpperCase();
    document.getElementById('dataset-name').textContent = data.dataset_name;
    document.getElementById('dataset-rows').textContent = `${data.dataset_rows} registros`;
}

function updateTrendMetrics(data) {
    const trendDir = data.trend_direction;
    const trendColor = trendDir === 'creciente' ? 'var(--success)' :
                       trendDir === 'decreciente' ? 'var(--danger)' :
                       trendDir === 'clasificacion' ? 'var(--primary)' :
                       trendDir === 'clustering' ? 'var(--warning)' : 'var(--muted)';

    const trendEl = document.getElementById('trend-direction');
    trendEl.textContent = formatTrendDirection(trendDir);
    trendEl.style.color = trendColor;

    document.getElementById('trend-rate').textContent = data.trend_rate_label;

    const confColor = data.confidence === 'alta' ? 'var(--success)' :
                      data.confidence === 'media' ? 'var(--warning)' : 'var(--danger)';
    const confEl = document.getElementById('confidence');
    confEl.textContent = data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
    confEl.style.color = confColor;
    document.getElementById('confidence-score').textContent = `Score: ${(data.confidence_score * 100).toFixed(1)}%`;
}

function formatTrendDirection(dir) {
    const labels = {
        'creciente': 'Tendencia Creciente',
        'decreciente': 'Tendencia Decreciente',
        'estable': 'Tendencia Estable',
        'clasificacion': 'Clasificacion',
        'clustering': 'Segmentacion',
        'no lineal': 'Patron No Lineal'
    };
    return labels[dir] || dir.charAt(0).toUpperCase() + dir.slice(1);
}

function updateInterpretation(data) {
    document.getElementById('interpretation').textContent = data.interpretation;

    const predNextEl = document.getElementById('prediction-next');
    if (data.model_type === 'kmeans') {
        predNextEl.textContent = 'N/A';
    } else if (data.model_type === 'logistic_regression') {
        predNextEl.textContent = `${(data.prediction_next * 100).toFixed(1)}%`;
    } else if (data.trend_direction === 'no lineal') {
        predNextEl.textContent = `MSE: ${data.trend_rate.toFixed(6)}`;
    } else {
        predNextEl.textContent = `$${data.prediction_next.toLocaleString('es', {maximumFractionDigits: 0})}`;
    }
}

function renderChart(data) {
    if (predictionChart) predictionChart.destroy();

    const ctx = document.getElementById('prediction-chart').getContext('2d');
    const chartTextColor = '#a1a1aa';
    const chartGridColor = '#2d2d44';

    const modelType = data.model_type;

    if (modelType === 'logistic_regression') {
        renderLogisticChart(ctx, data, chartTextColor, chartGridColor);
    } else if (modelType === 'kmeans') {
        renderKMeansChart(ctx, data, chartTextColor, chartGridColor);
    } else if (modelType === 'neural_network') {
        renderNeuralChart(ctx, data, chartTextColor, chartGridColor);
    } else {
        renderLinearChart(ctx, data, chartTextColor, chartGridColor);
    }
}

function renderLinearChart(ctx, data, chartTextColor, chartGridColor) {
    const historicalData = data.x_data.map((x, i) => ({ x, y: data.y_data[i] }));
    const fittedLine = data.x_data.map((x, i) => ({ x, y: data.predictions[i] }));
    const lastX = data.x_data[data.x_data.length - 1];
    const futureLine = data.future_x.map((x, i) => ({ x, y: data.future_predictions[i] }));

    predictionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Datos reales',
                    data: historicalData,
                    backgroundColor: 'rgba(0, 212, 255, 0.5)',
                    borderColor: '#00d4ff',
                    pointRadius: 4
                },
                {
                    label: 'Ajuste del modelo',
                    data: fittedLine,
                    type: 'line',
                    borderColor: '#e74c3c',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'Prediccion futura',
                    data: futureLine,
                    type: 'line',
                    borderColor: '#27ae60',
                    borderWidth: 2,
                    borderDash: [6, 4],
                    pointRadius: 4,
                    pointBackgroundColor: '#27ae60',
                    fill: false
                }
            ]
        },
        options: getChartOptions(data, chartTextColor, chartGridColor)
    });
}

function renderLogisticChart(ctx, data, chartTextColor, chartGridColor) {
    const scatterData = data.x_data.map((x, i) => ({ x, y: data.y_data[i] }));
    const predictions = data.x_data.map((x, i) => ({ x, y: data.predictions[i] }));

    predictionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Datos reales',
                    data: scatterData,
                    backgroundColor: data.y_data.map(y => y === 1 ? 'rgba(39, 174, 96, 0.6)' : 'rgba(231, 76, 60, 0.6)'),
                    borderColor: data.y_data.map(y => y === 1 ? '#27ae60' : '#e74c3c'),
                    pointRadius: 5
                },
                {
                    label: 'Predicciones',
                    data: predictions,
                    type: 'line',
                    borderColor: '#9b59b6',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false,
                    stepped: 'middle'
                }
            ]
        },
        options: getChartOptions(data, chartTextColor, chartGridColor)
    });
}

function renderKMeansChart(ctx, data, chartTextColor, chartGridColor) {
    const CLUSTER_COLORS = [
        'rgba(0, 212, 255, 0.7)',
        'rgba(231, 76, 60, 0.7)',
        'rgba(39, 174, 96, 0.7)',
        'rgba(243, 156, 18, 0.7)',
        'rgba(155, 89, 182, 0.7)'
    ];

    const uniquePredictions = [...new Set(data.predictions)].filter(p => p !== undefined);
    const datasets = uniquePredictions.map((cluster, idx) => ({
        label: `Cluster ${cluster}`,
        data: data.x_data.map((x, i) => ({ x, y: data.y_data[i] })).filter((_, i) => data.predictions[i] === cluster),
        backgroundColor: CLUSTER_COLORS[idx % CLUSTER_COLORS.length],
        pointRadius: 5
    }));

    predictionChart = new Chart(ctx, {
        type: 'scatter',
        data: { datasets },
        options: getChartOptions(data, chartTextColor, chartGridColor)
    });
}

function renderNeuralChart(ctx, data, chartTextColor, chartGridColor) {
    const historicalData = data.x_data.map((x, i) => ({ x, y: data.y_data[i] }));
    const predictions = data.x_data.map((x, i) => ({ x, y: data.predictions[i] }));

    predictionChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Datos reales',
                    data: historicalData,
                    backgroundColor: 'rgba(0, 212, 255, 0.5)',
                    borderColor: '#00d4ff',
                    pointRadius: 4
                },
                {
                    label: 'Predicciones NN',
                    data: predictions,
                    type: 'line',
                    borderColor: '#f39c12',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: getChartOptions(data, chartTextColor, chartGridColor)
    });
}

function getChartOptions(data, chartTextColor, chartGridColor) {
    return {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { labels: { color: chartTextColor } }
        },
        scales: {
            x: {
                title: { display: true, text: data.x_col, color: chartTextColor },
                ticks: { color: chartTextColor },
                grid: { color: chartGridColor }
            },
            y: {
                title: { display: true, text: data.y_col, color: chartTextColor },
                ticks: { color: chartTextColor },
                grid: { color: chartGridColor }
            }
        }
    };
}
