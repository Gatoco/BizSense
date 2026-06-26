const API = 'http://localhost:5000';

let history = [];
let xData = [];
let yData = [];
let currentStep = 0;
let isPlaying = false;
let animationTimer = null;
let regressionChart = null;
let costChart = null;

document.addEventListener('DOMContentLoaded', async () => {
    await loadDatasets();
    await loadConfig();

    document.getElementById('dataset-select').addEventListener('change', async (e) => {
        const id = e.target.value;
        if (!id) return;
        await loadDatasetColumns(parseInt(id));
    });

    document.getElementById('train-btn').addEventListener('click', train);
    document.getElementById('play-btn').addEventListener('click', startAnimation);
    document.getElementById('pause-btn').addEventListener('click', stopAnimation);
    document.getElementById('reset-btn').addEventListener('click', reset);
    document.getElementById('step-btn').addEventListener('click', stepForward);
    document.getElementById('speed-slider').addEventListener('input', changeSpeed);
});

async function loadDatasets() {
    try {
        const res = await fetch(`${API}/datasets`);
        const datasets = await res.json();
        const select = document.getElementById('dataset-select');

        if (datasets.length === 0) {
            document.getElementById('config-form').classList.add('hidden');
            document.getElementById('no-datasets').classList.remove('hidden');
            return;
        }

        select.innerHTML = '<option value="">Selecciona un dataset...</option>';
        datasets.forEach(ds => {
            select.add(new Option(`${ds.name} (${ds.rows} filas)`, ds.id));
        });
    } catch (err) {
        console.error('Error loading datasets:', err);
    }
}

async function loadConfig() {
    try {
        const res = await fetch(`${API}/config`);
        const config = await res.json();
        document.getElementById('alpha').value = config.default_alpha;
        document.getElementById('iterations').value = config.default_iterations;
    } catch (err) {
        console.error('Error loading config:', err);
    }
}

async function loadDatasetColumns(id) {
    try {
        const res = await fetch(`${API}/datasets/${id}`);
        const data = await res.json();

        const xCol = document.getElementById('x-col');
        const yCol = document.getElementById('y-col');
        xCol.innerHTML = '';
        yCol.innerHTML = '';

        data.columns.forEach(col => {
            xCol.add(new Option(col, col));
            yCol.add(new Option(col, col));
        });

        if (data.columns.length > 1) {
            yCol.selectedIndex = 1;
        }
    } catch (err) {
        console.error('Error loading columns:', err);
    }
}

async function train() {
    const datasetId = parseInt(document.getElementById('dataset-select').value);
    if (!datasetId) {
        alert('Selecciona un dataset');
        return;
    }

    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Entrenando...';

    try {
        const res = await fetch(`${API}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_id: datasetId,
                x_col: document.getElementById('x-col').value,
                y_col: document.getElementById('y-col').value,
                alpha: parseFloat(document.getElementById('alpha').value),
                iterations: parseInt(document.getElementById('iterations').value)
            })
        });

        const data = await res.json();
        history = data.history;
        xData = data.x_data;
        yData = data.y_data;
        currentStep = 0;

        initCharts();
        document.getElementById('dashboard').classList.remove('hidden');
        document.getElementById('results').classList.add('hidden');
        renderStep(0);
    } catch (err) {
        alert('Error al entrenar modelo.');
        console.error(err);
    } finally {
        trainBtn.disabled = false;
        trainBtn.textContent = 'Entrenar modelo';
    }
}

function initCharts() {
    if (regressionChart) regressionChart.destroy();
    if (costChart) costChart.destroy();

    const scatterData = xData.map((x, i) => ({ x, y: yData[i] }));
    const xColName = document.getElementById('x-col').value;
    const yColName = document.getElementById('y-col').value;

    const chartTextColor = '#a1a1aa';
    const chartGridColor = '#2d2d44';

    regressionChart = new Chart(document.getElementById('regression-chart'), {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Datos',
                    data: scatterData,
                    backgroundColor: 'rgba(0, 212, 255, 0.5)',
                    borderColor: '#00d4ff',
                    pointRadius: 4
                },
                {
                    label: 'Regresion',
                    data: [],
                    type: 'line',
                    borderColor: '#e74c3c',
                    borderWidth: 3,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: chartTextColor } }
            },
            scales: {
                x: {
                    title: { display: true, text: xColName, color: chartTextColor },
                    ticks: { color: chartTextColor },
                    grid: { color: chartGridColor }
                },
                y: {
                    title: { display: true, text: yColName, color: chartTextColor },
                    ticks: { color: chartTextColor },
                    grid: { color: chartGridColor }
                }
            },
            animation: { duration: 0 }
        }
    });

    costChart = new Chart(document.getElementById('cost-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'J(theta)',
                data: [],
                borderColor: '#27ae60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                borderWidth: 2,
                fill: true,
                pointRadius: 2
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { labels: { color: chartTextColor } }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Iteracion', color: chartTextColor },
                    ticks: { color: chartTextColor },
                    grid: { color: chartGridColor }
                },
                y: {
                    title: { display: true, text: 'Costo', color: chartTextColor },
                    ticks: { color: chartTextColor },
                    grid: { color: chartGridColor }
                }
            },
            animation: { duration: 0 }
        }
    });
}

function renderStep(stepIndex) {
    if (stepIndex < 0 || stepIndex >= history.length) return;

    const step = history[stepIndex];

    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const lineData = [
        { x: xMin, y: step.theta_0_real + step.theta_1_real * xMin },
        { x: xMax, y: step.theta_0_real + step.theta_1_real * xMax }
    ];
    regressionChart.data.datasets[1].data = lineData;
    regressionChart.update();

    costChart.data.labels = history.slice(0, stepIndex + 1).map(h => h.iteration);
    costChart.data.datasets[0].data = history.slice(0, stepIndex + 1).map(h => h.cost);
    costChart.update();

    updateFormulas(step);
}

function updateFormulas(step) {
    document.getElementById('iteration-info').textContent =
        `Iteracion ${step.iteration} de ${history.length}`;

    document.getElementById('formulas').innerHTML = `
        <div class="formula-row">
            <strong>Hipotesis:</strong>
            \\( h(x) = ${step.theta_0_real.toFixed(4)} + ${step.theta_1_real.toFixed(4)} \\cdot x \\)
        </div>
        <div class="formula-row">
            <strong>Funcion de costo:</strong>
            \\( J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m}(h(x_i) - y_i)^2 = ${step.cost.toFixed(4)} \\)
        </div>
        <div class="formula-row">
            <strong>Derivadas parciales (gradiente):</strong>
            \\( \\frac{\\partial J}{\\partial \\theta_0} = \\frac{1}{m} \\sum_{i=1}^{m}(h(x_i) - y_i) = ${step.gradient_0.toFixed(6)} \\)<br>
            \\( \\frac{\\partial J}{\\partial \\theta_1} = \\frac{1}{m} \\sum_{i=1}^{m}(h(x_i) - y_i) \\cdot x_i = ${step.gradient_1.toFixed(6)} \\)
        </div>
        <div class="formula-row">
            <strong>Actualizacion de parametros:</strong>
            \\( \\theta_0 := \\theta_0 - \\alpha \\cdot \\frac{\\partial J}{\\partial \\theta_0} = ${step.theta_0_real.toFixed(4)} \\)<br>
            \\( \\theta_1 := \\theta_1 - \\alpha \\cdot \\frac{\\partial J}{\\partial \\theta_1} = ${step.theta_1_real.toFixed(4)} \\)
        </div>
    `;

    if (window.MathJax && window.MathJax.typeset) {
        MathJax.typeset();
    }
}

function animateStep() {
    if (currentStep >= history.length - 1) {
        stopAnimation();
        showResults();
        return;
    }
    currentStep++;
    renderStep(currentStep);
}

function startAnimation() {
    if (isPlaying) return;
    isPlaying = true;
    const speed = parseInt(document.getElementById('speed-slider').value);
    animationTimer = setInterval(animateStep, speed);
}

function stopAnimation() {
    isPlaying = false;
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }
}

function reset() {
    stopAnimation();
    currentStep = 0;
    document.getElementById('results').classList.add('hidden');
    renderStep(0);
}

function stepForward() {
    stopAnimation();
    if (currentStep < history.length - 1) {
        currentStep++;
        renderStep(currentStep);
        if (currentStep === history.length - 1) {
            showResults();
        }
    }
}

function changeSpeed(e) {
    document.getElementById('speed-value').textContent = `${e.target.value}ms`;
    if (isPlaying) {
        stopAnimation();
        startAnimation();
    }
}

function showResults() {
    const final = history[history.length - 1];
    document.getElementById('results').classList.remove('hidden');
    document.getElementById('final-values').innerHTML = `
        <div class="result-card">
            <div class="label">theta_0 (intercepto)</div>
            <div class="value">${final.theta_0_real.toFixed(4)}</div>
        </div>
        <div class="result-card">
            <div class="label">theta_1 (pendiente)</div>
            <div class="value">${final.theta_1_real.toFixed(4)}</div>
        </div>
        <div class="result-card">
            <div class="label">Costo final J(theta)</div>
            <div class="value">${final.cost.toFixed(4)}</div>
        </div>
        <div class="result-card">
            <div class="label">Ecuacion final</div>
            <div class="value" style="font-size: 1rem;">y = ${final.theta_0_real.toFixed(2)} + ${final.theta_1_real.toFixed(2)}x</div>
        </div>
    `;
}