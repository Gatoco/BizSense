const API = 'http://localhost:5000';

let history = [];
let xData = [];
let yData = [];
let currentStep = 0;
let isPlaying = false;
let animationTimer = null;
let mathJaxTimer = null;
let regressionChart = null;
let costChart = null;
let currentModel = 'linear_regression';
let currentAlpha = 0.01;
let currentK = 3;

const CLUSTER_COLORS = [
    'rgba(220, 220, 220, 0.8)',
    'rgba(160, 160, 160, 0.8)',
    'rgba(190, 190, 190, 0.8)',
    'rgba(130, 130, 130, 0.8)',
    'rgba(200, 200, 200, 0.8)',
    'rgba(110, 110, 110, 0.8)',
    'rgba(170, 170, 170, 0.8)',
    'rgba(90, 90, 90, 0.8)',
    'rgba(210, 210, 210, 0.8)',
    'rgba(140, 140, 140, 0.8)'
];

document.addEventListener('DOMContentLoaded', async () => {
    await loadDatasets();
    await loadConfig();

    document.getElementById('dataset-select').addEventListener('change', async (e) => {
        const id = e.target.value;
        if (!id) return;
        await loadDatasetColumns(parseInt(id));
    });

    document.getElementById('model-select').addEventListener('change', (e) => {
        currentModel = e.target.value;
        const kRow = document.getElementById('k-row');
        if (currentModel === 'kmeans') {
            kRow.classList.remove('hidden');
        } else {
            kRow.classList.add('hidden');
        }
    });

    document.getElementById('train-btn').addEventListener('click', train);
    document.getElementById('play-btn').addEventListener('click', startAnimation);
    document.getElementById('pause-btn').addEventListener('click', stopAnimation);
    document.getElementById('reset-btn').addEventListener('click', reset);
    document.getElementById('step-btn').addEventListener('click', stepForward);
    document.getElementById('step-back-btn').addEventListener('click', stepBackward);
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

    currentModel = document.getElementById('model-select').value;
    const trainBtn = document.getElementById('train-btn');
    trainBtn.disabled = true;
    trainBtn.textContent = 'Entrenando...';

    try {
        const body = {
            dataset_id: datasetId,
            x_col: document.getElementById('x-col').value,
            y_col: document.getElementById('y-col').value,
            alpha: parseFloat(document.getElementById('alpha').value),
            iterations: parseInt(document.getElementById('iterations').value),
            model_type: currentModel
        };

        currentAlpha = body.alpha;

        if (currentModel === 'kmeans') {
            body.k = parseInt(document.getElementById('k-clusters').value);
            currentK = body.k;
        }

        const res = await fetch(`${API}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
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

    const xColName = document.getElementById('x-col').value;
    const yColName = document.getElementById('y-col').value;
    const chartTextColor = '#808080';
    const chartGridColor = '#3a3a3a';

    const baseOptions = {
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
    };

    if (currentModel === 'kmeans') {
        const k = parseInt(document.getElementById('k-clusters').value);
        const datasets = [];

        for (let c = 0; c < k; c++) {
            datasets.push({
                label: `Cluster ${c + 1}`,
                data: [],
                backgroundColor: CLUSTER_COLORS[c % CLUSTER_COLORS.length],
                pointRadius: 5
            });
        }

        datasets.push({
            label: 'Centroides',
            data: [],
            backgroundColor: 'rgba(255, 255, 255, 0.85)',
            borderColor: '#e0e0e0',
            pointRadius: 10,
            pointStyle: 'crossRot',
            borderWidth: 2
        });

        regressionChart = new Chart(document.getElementById('regression-chart'), {
            type: 'scatter',
            data: { datasets },
            options: baseOptions
        });
    } else {
        const scatterData = xData.map((x, i) => ({ x, y: yData[i] }));

        regressionChart = new Chart(document.getElementById('regression-chart'), {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Datos',
                        data: scatterData,
                        backgroundColor: 'rgba(200, 200, 200, 0.5)',
                        borderColor: '#b0b0b0',
                        pointRadius: 4
                    },
                    {
                        label: 'Modelo',
                        data: [],
                        type: 'line',
                        borderColor: '#a04040',
                        borderWidth: 3,
                        pointRadius: 0,
                        fill: false
                    }
                ]
            },
            options: baseOptions
        });
    }

    const costLabel = currentModel === 'kmeans' ? 'Inercia' :
                      currentModel === 'neural_network' ? 'MSE' : 'J(theta)';

    costChart = new Chart(document.getElementById('cost-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: costLabel,
                data: [],
                borderColor: '#909090',
                backgroundColor: 'rgba(144, 144, 144, 0.1)',
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

    if (currentModel === 'kmeans') {
        renderKMeansStep(step);
    } else if (currentModel === 'logistic_regression') {
        renderLogisticStep(step);
    } else if (currentModel === 'neural_network') {
        renderNeuralStep(step);
    } else {
        renderLinearStep(step);
    }

    costChart.data.labels = history.slice(0, stepIndex + 1).map(h => h.iteration);
    costChart.data.datasets[0].data = history.slice(0, stepIndex + 1).map(h => h.cost);
    costChart.update();

    updateFormulas(step);
}

function renderLinearStep(step) {
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const lineData = [
        { x: xMin, y: step.theta_0_real + step.theta_1_real * xMin },
        { x: xMax, y: step.theta_0_real + step.theta_1_real * xMax }
    ];
    regressionChart.data.datasets[1].data = lineData;
    regressionChart.update();
}

function renderLogisticStep(step) {
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData);
    const points = [];
    for (let x = xMin; x <= xMax; x += (xMax - xMin) / 100) {
        const z = step.theta_0_real + step.theta_1_real * x;
        const sigmoid = 1 / (1 + Math.exp(-z));
        points.push({ x, y: sigmoid });
    }
    regressionChart.data.datasets[1].data = points;
    regressionChart.update();
}

function renderKMeansStep(step) {
    const extra = step.extra || {};
    const assignments = extra.assignments || [];
    const centroids = extra.centroids || [];
    const k = extra.k || 3;

    for (let c = 0; c < k; c++) {
        regressionChart.data.datasets[c].data = [];
    }

    xData.forEach((x, i) => {
        const cluster = assignments[i] || 0;
        if (cluster < k) {
            regressionChart.data.datasets[cluster].data.push({ x, y: yData[i] });
        }
    });

    regressionChart.data.datasets[k].data = centroids.map(c => ({ x: c[0], y: c[1] }));
    regressionChart.update();
}

function renderNeuralStep(step) {
    const extra = step.extra || {};
    const predictions = extra.predictions || step.predictions || [];
    const lineData = xData.map((x, i) => ({ x, y: predictions[i] }));
    regressionChart.data.datasets[1].data = lineData;
    regressionChart.update();
}

function formulaBlock(label, generalLatex, computedLatex) {
    const generalParts = generalLatex.split(/\\\\?\[?\d*pt\]?/).filter(s => s.trim());
    const computedParts = computedLatex.split(/\\\\?\[?\d*pt\]?/).filter(s => s.trim());

    const generalHtml = generalParts.map(p => `\\( ${p.trim()} \\)`).join('<br>');
    const computedHtml = computedParts.map(p => `\\( ${p.trim()} \\)`).join('<br>');

    return `
        <div class="formula-row">
            <strong>${label}</strong>
            <div class="formula-general">${generalHtml}</div>
            <div class="formula-separator"></div>
            <div class="formula-computed">${computedHtml}</div>
        </div>
    `;
}

function updateFormulas(step) {
    document.getElementById('iteration-info').textContent =
        `Iteracion ${step.iteration} de ${history.length}`;

    const m = xData.length;
    const a = currentAlpha;
    let html = '';

    if (currentModel === 'linear_regression') {
        const t0 = step.theta_0_real;
        const t1 = step.theta_1_real;
        const g0 = step.gradient_0;
        const g1 = step.gradient_1;
        const t0_old = t0 + a * g0;
        const t1_old = t1 + a * g1;
        const sgn1 = t1 >= 0 ? '+' : '-';
        const sgn1o = t1_old >= 0 ? '+' : '-';
        html = `
            ${formulaBlock(
                'Hipotesis',
                'h(x) = \\theta_0 + \\theta_1 \\cdot x',
                `h(x) = ${t0.toFixed(4)} ${sgn1} ${Math.abs(t1).toFixed(4)} \\cdot x`
            )}
            ${formulaBlock(
                'Funcion de costo (MSE)',
                'J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m}(h(x_i) - y_i)^2',
                `J(\\theta) = \\frac{1}{2 \\cdot ${m}} \\sum_{i=1}^{${m}}(h(x_i) - y_i)^2 = ${step.cost.toFixed(4)}`
            )}
            ${formulaBlock(
                'Derivadas parciales (gradiente)',
                '\\frac{\\partial J}{\\partial \\theta_0} = \\frac{1}{m} \\sum_{i=1}^{m}(h(x_i) - y_i), \\quad \\frac{\\partial J}{\\partial \\theta_1} = \\frac{1}{m} \\sum_{i=1}^{m}(h(x_i) - y_i) \\cdot x_i',
                `\\frac{\\partial J}{\\partial \\theta_0} = \\frac{1}{${m}} \\sum_{i=1}^{${m}}(h(x_i) - y_i) = ${g0.toFixed(6)} \\\\[4pt] \\frac{\\partial J}{\\partial \\theta_1} = \\frac{1}{${m}} \\sum_{i=1}^{${m}}(h(x_i) - y_i) \\cdot x_i = ${g1.toFixed(6)}`
            )}
            ${formulaBlock(
                'Actualizacion de parametros',
                '\\theta_j := \\theta_j - \\alpha \\cdot \\frac{\\partial J}{\\partial \\theta_j}',
                `\\theta_0 := ${t0_old.toFixed(4)} - ${a} \\cdot ${g0.toFixed(6)} = ${t0.toFixed(4)} \\\\[4pt] \\theta_1 := ${t1_old.toFixed(4)} - ${a} \\cdot ${g1.toFixed(6)} = ${t1.toFixed(4)}`
            )}
        `;
    } else if (currentModel === 'logistic_regression') {
        const t0 = step.theta_0_real;
        const t1 = step.theta_1_real;
        const g0 = step.gradient_0;
        const g1 = step.gradient_1;
        const t0_old = t0 + a * g0;
        const t1_old = t1 + a * g1;
        const sgn = t1 >= 0 ? '+' : '-';
        const sgnOld = t1_old >= 0 ? '+' : '-';
        html = `
            ${formulaBlock(
                'Funcion sigmoide',
                'g(z) = \\frac{1}{1 + e^{-z}}',
                `g(z) = \\frac{1}{1 + e^{-z}}, \\quad z = ${t0.toFixed(4)} ${sgn} ${Math.abs(t1).toFixed(4)} \\cdot x`
            )}
            ${formulaBlock(
                'Hipotesis',
                'h(x) = g(\\theta_0 + \\theta_1 x) = \\frac{1}{1 + e^{-(\\theta_0 + \\theta_1 x)}}',
                `h(x) = \\frac{1}{1 + e^{-(${t0.toFixed(4)} ${sgn} ${Math.abs(t1).toFixed(4)} \\cdot x)}}`
            )}
            ${formulaBlock(
                'Funcion de costo (cross-entropy)',
                'J(\\theta) = -\\frac{1}{m} \\sum_{i=1}^{m}[y_i \\log(h(x_i)) + (1-y_i) \\log(1-h(x_i))]',
                `J(\\theta) = -\\frac{1}{${m}} \\sum_{i=1}^{${m}}[y_i \\log(h(x_i)) + (1-y_i) \\log(1-h(x_i))] = ${step.cost.toFixed(4)}`
            )}
            ${formulaBlock(
                'Derivadas',
                '\\frac{\\partial J}{\\partial \\theta_0} = \\frac{1}{m} \\sum(h(x_i) - y_i), \\quad \\frac{\\partial J}{\\partial \\theta_1} = \\frac{1}{m} \\sum(h(x_i) - y_i) \\cdot x_i',
                `\\frac{\\partial J}{\\partial \\theta_0} = \\frac{1}{${m}} \\sum_{i=1}^{${m}}(h(x_i) - y_i) = ${g0.toFixed(6)} \\\\[4pt] \\frac{\\partial J}{\\partial \\theta_1} = \\frac{1}{${m}} \\sum_{i=1}^{${m}}(h(x_i) - y_i) \\cdot x_i = ${g1.toFixed(6)}`
            )}
            ${formulaBlock(
                'Actualizacion de parametros',
                '\\theta_j := \\theta_j - \\alpha \\cdot \\frac{\\partial J}{\\partial \\theta_j}',
                `\\theta_0 := ${t0_old.toFixed(4)} - ${a} \\cdot ${g0.toFixed(6)} = ${t0.toFixed(4)} \\\\[4pt] \\theta_1 := ${t1_old.toFixed(4)} - ${a} \\cdot ${g1.toFixed(6)} = ${t1.toFixed(4)}`
            )}
            <div class="formula-row">
                <strong>Limites de la sigmoide</strong>
                <div class="formula-general">\\( g(z) \\to 1 \\) cuando \\( z \\to +\\infty \\), \\quad g(z) \\to 0 \\) cuando \\( z \\to -\\infty \\)</div>
            </div>
        `;
    } else if (currentModel === 'kmeans') {
        const extra = step.extra || {};
        const inertia = extra.inertia || step.gradient_0;
        const converged = extra.converged;
        const centroids = extra.centroids || [];
        const assignments = extra.assignments || [];
        const counts = {};
        assignments.forEach(c => { counts[c] = (counts[c] || 0) + 1; });
        const clusterCounts = Object.values(counts).join(',\\ ');
        const centroidsStr = centroids.map((c, j) => `\\mu_{${j}} = (${c[0].toFixed(2)}, ${c[1].toFixed(2)})`).join(',\\ ');
        html = `
            ${formulaBlock(
                'Asignacion al cluster mas cercano',
                'c^{(i)} = \\arg\\min_j \\|x^{(i)} - \\mu_j\\|^2',
                `J = \\sum_{i=1}^{${m}} \\|x^{(i)} - \\mu_{c^{(i)}}\\|^2 = ${inertia.toFixed(4)}`
            )}
            ${formulaBlock(
                'Actualizacion de centroides',
                '\\mu_j = \\frac{1}{|C_j|} \\sum_{i \\in C_j} x^{(i)}',
                `${centroidsStr} \\\\[4pt] |C_j| = [${clusterCounts}]`
            )}
            <div class="formula-row">
                <strong>Limite (convergencia)</strong>
                <div class="formula-general">\\( n \\to \\infty \\): las asignaciones y centroides dejan de cambiar.</div>
                <div class="formula-separator"></div>
                <div class="formula-computed">${converged ? 'Convergio en esta iteracion.' : 'Aun iterando...'}</div>
            </div>
        `;
    } else if (currentModel === 'neural_network') {
        const extra = step.extra || {};
        const weightsNorm = extra.weights_norm || 0;
        const b2 = step.theta_0;
        const w2_0 = step.theta_1;
        const gradMag = step.gradient_0;
        const d1norm = step.gradient_1;
        html = `
            ${formulaBlock(
                'Forward propagation',
                'z^{(1)} = W^{(1)} x + b^{(1)}, \\quad a^{(1)} = g(z^{(1)}) \\\\[6pt] z^{(2)} = W^{(2)} a^{(1)} + b^{(2)}, \\quad a^{(2)} = g(z^{(2)})',
                `b^{(2)} = ${b2.toFixed(4)}, \\quad W^{(2)}_0 = ${w2_0.toFixed(4)} \\\\[6pt] \\|W\\| = ${weightsNorm.toFixed(4)}`
            )}
            ${formulaBlock(
                'Backpropagation (regla de la cadena)',
                '\\delta^{(2)} = a^{(2)} - y, \\quad \\delta^{(1)} = (W^{(2)})^T \\delta^{(2)} \\cdot g\'(z^{(1)})',
                `\\delta^{(2)} = a^{(2)} - y \\\\[4pt] \\delta^{(1)} = (W^{(2)})^T \\delta^{(2)} \\cdot g\'(z^{(1)}), \\quad \\|\\delta^{(1)}\\| = ${d1norm.toFixed(6)}`
            )}
            ${formulaBlock(
                'Funcion de costo (MSE)',
                'J = \\frac{1}{2m} \\sum_{i=1}^{m}(a^{(2)}_i - y_i)^2',
                `J = \\frac{1}{2 \\cdot ${m}} \\sum_{i=1}^{${m}}(a^{(2)}_i - y_i)^2 = ${step.cost.toFixed(6)}`
            )}
            ${formulaBlock(
                'Gradientes (regla de la cadena)',
                '\\frac{\\partial J}{\\partial W^{(2)}} = \\delta^{(2)} (a^{(1)})^T, \\quad \\frac{\\partial J}{\\partial W^{(1)}} = \\delta^{(1)} x^T',
                `\\frac{\\partial J}{\\partial W^{(2)}} = \\delta^{(2)} (a^{(1)})^T \\\\[4pt] \\frac{\\partial J}{\\partial W^{(1)}} = \\delta^{(1)} x^T \\\\[4pt] \\text{magnitud del gradiente} = ${gradMag.toFixed(6)}`
            )}
        `;
    }

    document.getElementById('formulas').innerHTML = html;

    clearTimeout(mathJaxTimer);
    mathJaxTimer = setTimeout(() => {
        if (window.MathJax && window.MathJax.typeset) {
            MathJax.typeset();
        }
    }, 150);
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

function stepBackward() {
    stopAnimation();
    if (currentStep > 0) {
        currentStep--;
        document.getElementById('results').classList.add('hidden');
        renderStep(currentStep);
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

    let html = '';

    if (currentModel === 'linear_regression') {
        html = `
            <div class="result-card">
                <div class="label">theta_0 (intercepto)</div>
                <div class="value">${final.theta_0_real.toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">theta_1 (pendiente)</div>
                <div class="value">${final.theta_1_real.toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">Costo final</div>
                <div class="value">${final.cost.toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">Ecuacion</div>
                <div class="value" style="font-size: 1rem;">y = ${final.theta_0_real.toFixed(2)} + ${final.theta_1_real.toFixed(2)}x</div>
            </div>
        `;
    } else if (currentModel === 'logistic_regression') {
        html = `
            <div class="result-card">
                <div class="label">theta_0</div>
                <div class="value">${final.theta_0_real.toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">theta_1</div>
                <div class="value">${final.theta_1_real.toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">Costo final</div>
                <div class="value">${final.cost.toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">Ecuacion</div>
                <div class="value" style="font-size: 1rem;">g(${final.theta_0_real.toFixed(2)} + ${final.theta_1_real.toFixed(2)}x)</div>
            </div>
        `;
    } else if (currentModel === 'kmeans') {
        const extra = final.extra || {};
        html = `
            <div class="result-card">
                <div class="label">Clusters</div>
                <div class="value">${extra.k || 3}</div>
            </div>
            <div class="result-card">
                <div class="label">Inercia final</div>
                <div class="value">${(extra.inertia || final.gradient_0).toFixed(4)}</div>
            </div>
            <div class="result-card">
                <div class="label">Iteraciones</div>
                <div class="value">${final.iteration}</div>
            </div>
            <div class="result-card">
                <div class="label">Convergio</div>
                <div class="value" style="font-size: 1rem;">${extra.converged ? 'Si' : 'No'}</div>
            </div>
        `;
    } else if (currentModel === 'neural_network') {
        const extra = final.extra || {};
        html = `
            <div class="result-card">
                <div class="label">Costo final (MSE)</div>
                <div class="value">${final.cost.toFixed(6)}</div>
            </div>
            <div class="result-card">
                <div class="label">Iteraciones</div>
                <div class="value">${final.iteration}</div>
            </div>
            <div class="result-card">
                <div class="label">Neuronas ocultas</div>
                <div class="value">${extra.hidden_size || 4}</div>
            </div>
            <div class="result-card">
                <div class="label">Norma de pesos</div>
                <div class="value">${(extra.weights_norm || 0).toFixed(4)}</div>
            </div>
        `;
    }

    document.getElementById('final-values').innerHTML = html;
}