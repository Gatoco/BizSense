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

        const trendDir = data.trend_direction;
        const trendColor = trendDir === 'creciente' ? 'var(--success)' : trendDir === 'decreciente' ? 'var(--danger)' : 'var(--muted)';

        document.getElementById('trend-direction').textContent = trendDir.charAt(0).toUpperCase() + trendDir.slice(1);
        document.getElementById('trend-direction').style.color = trendColor;
        document.getElementById('trend-rate').textContent = data.trend_rate_label;

        document.getElementById('prediction-next').textContent = `$${data.prediction_next.toLocaleString('es', {maximumFractionDigits: 0})}`;

        document.getElementById('confidence').textContent = data.confidence.charAt(0).toUpperCase() + data.confidence.slice(1);
        const confColor = data.confidence === 'alta' ? 'var(--success)' : data.confidence === 'media' ? 'var(--warning)' : 'var(--danger)';
        document.getElementById('confidence').style.color = confColor;
        document.getElementById('confidence-score').textContent = `R² = ${data.confidence_score.toFixed(2)}`;

        document.getElementById('interpretation').textContent = data.interpretation;
        document.getElementById('model-type').textContent = data.model_type;
        document.getElementById('dataset-name').textContent = data.dataset_name;
        document.getElementById('dataset-rows').textContent = `${data.dataset_rows} registros`;

        renderPredictionChart(data);
    } catch (err) {
        console.error('Error loading insights:', err);
    }
}

function renderPredictionChart(data) {
    if (predictionChart) predictionChart.destroy();

    const ctx = document.getElementById('prediction-chart').getContext('2d');

    const historicalData = data.x_data.map((x, i) => ({ x, y: data.y_data[i] }));
    const fittedLine = data.x_data.map((x, i) => ({ x, y: data.predictions[i] }));

    const lastX = data.x_data[data.x_data.length - 1];
    const futureLine = data.future_x.map((x, i) => ({ x, y: data.future_predictions[i] }));

    const chartTextColor = '#a1a1aa';
    const chartGridColor = '#2d2d44';

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
        options: {
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
        }
    });
}