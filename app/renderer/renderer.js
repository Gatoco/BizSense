const API = 'http://localhost:5000';

let history = [];
let xData = [];
let yData = [];
let currentStep = 0;
let isPlaying = false;
let animationTimer = null;
let regressionChart = null;
let costChart = null;

const csvInput = document.getElementById('csv-input');
const previewContainer = document.getElementById('preview-container');
const previewTable = document.getElementById('preview-table');
const rowCount = document.getElementById('row-count');
const columnSelector = document.getElementById('column-selector');
const xColSelect = document.getElementById('x-col');
const yColSelect = document.getElementById('y-col');
const alphaInput = document.getElementById('alpha');
const iterationsInput = document.getElementById('iterations');
const trainBtn = document.getElementById('train-btn');
const dashboard = document.getElementById('dashboard');
const playBtn = document.getElementById('play-btn');
const pauseBtn = document.getElementById('pause-btn');
const resetBtn = document.getElementById('reset-btn');
const stepBtn = document.getElementById('step-btn');
const speedSlider = document.getElementById('speed-slider');
const speedValue = document.getElementById('speed-value');

csvInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const res = await fetch(`${API}/upload`, { method: 'POST', body: formData });
        const data = await res.json();
        
        xColSelect.innerHTML = '';
        yColSelect.innerHTML = '';
        data.columns.forEach(col => {
            xColSelect.add(new Option(col, col));
            yColSelect.add(new Option(col, col));
        });
        
        if (data.columns.length > 1) {
            yColSelect.selectedIndex = 1;
        }
        
        let tableHtml = '<table><thead><tr>';
        data.columns.forEach(col => tableHtml += `<th>${col}</th>`);
        tableHtml += '</tr></thead><tbody>';
        data.preview.forEach(row => {
            tableHtml += '<tr>';
            data.columns.forEach(col => {
                const val = row[col];
                tableHtml += `<td>${val !== null && val !== undefined ? val : '-'}</td>`;
            });
            tableHtml += '</tr>';
        });
        tableHtml += '</tbody></table>';
        previewTable.innerHTML = tableHtml;
        rowCount.textContent = `Total: ${data.rows} filas`;
        
        previewContainer.hidden = false;
        columnSelector.hidden = false;
    } catch (err) {
        alert('Error al subir archivo. Asegúrate de que el servidor Python esté corriendo.');
        console.error(err);
    }
});

trainBtn.addEventListener('click', async () => {
    trainBtn.disabled = true;
    trainBtn.textContent = 'Entrenando...';
    
    try {
        const res = await fetch(`${API}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x_col: xColSelect.value,
                y_col: yColSelect.value,
                alpha: parseFloat(alphaInput.value),
                iterations: parseInt(iterationsInput.value)
            })
        });
        
        const data = await res.json();
        history = data.history;
        xData = data.x_data;
        yData = data.y_data;
        currentStep = 0;
        
        initCharts();
        dashboard.hidden = false;
        renderStep(0);
    } catch (err) {
        alert('Error al entrenar modelo.');
        console.error(err);
    } finally {
        trainBtn.disabled = false;
        trainBtn.textContent = 'Entrenar modelo';
    }
});

function initCharts() {
    if (regressionChart) regressionChart.destroy();
    if (costChart) costChart.destroy();
    
    const scatterData = xData.map((x, i) => ({ x, y: yData[i] }));
    
    regressionChart = new Chart(document.getElementById('regression-chart'), {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Datos',
                    data: scatterData,
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    borderColor: '#667eea',
                    pointRadius: 4
                },
                {
                    label: 'Regresión',
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
            scales: {
                x: { title: { display: true, text: xColSelect.value } },
                y: { title: { display: true, text: yColSelect.value } }
            },
            animation: { duration: 0 }
        }
    });
    
    costChart = new Chart(document.getElementById('cost-chart'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'J(θ)',
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
            scales: {
                x: { title: { display: true, text: 'Iteración' } },
                y: { title: { display: true, text: 'Costo' } }
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
        `Iteración ${step.iteration} de ${history.length}`;
    
    document.getElementById('formulas').innerHTML = `
        <div class="formula-row">
            <strong>Hipótesis:</strong><br>
            \\( h(x) = ${step.theta_0_real.toFixed(4)} + ${step.theta_1_real.toFixed(4)} \\cdot x \\)
        </div>
        <div class="formula-row">
            <strong>Función de costo:</strong><br>
            \\( J(\\theta) = \\frac{1}{2m} \\sum_{i=1}^{m}(h(x_i) - y_i)^2 = ${step.cost.toFixed(4)} \\)
        </div>
        <div class="formula-row">
            <strong>Derivadas parciales (gradiente):</strong><br>
            \\( \\frac{\\partial J}{\\partial \\theta_0} = \\frac{1}{m} \\sum_{i=1}^{m}(h(x_i) - y_i) = ${step.gradient_0.toFixed(6)} \\)<br>
            \\( \\frac{\\partial J}{\\partial \\theta_1} = \\frac{1}{m} \\sum_{i=1}^{m}(h(x_i) - y_i) \\cdot x_i = ${step.gradient_1.toFixed(6)} \\)
        </div>
        <div class="formula-row">
            <strong>Actualización de parámetros:</strong><br>
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
    const speed = parseInt(speedSlider.value);
    animationTimer = setInterval(animateStep, speed);
}

function stopAnimation() {
    isPlaying = false;
    if (animationTimer) {
        clearInterval(animationTimer);
        animationTimer = null;
    }
}

function showResults() {
    const final = history[history.length - 1];
    document.getElementById('results').hidden = false;
    document.getElementById('final-values').innerHTML = `
        <div class="final-value-card">
            <div class="label">θ₀ (intercepto)</div>
            <div class="value">${final.theta_0_real.toFixed(4)}</div>
        </div>
        <div class="final-value-card">
            <div class="label">θ₁ (pendiente)</div>
            <div class="value">${final.theta_1_real.toFixed(4)}</div>
        </div>
        <div class="final-value-card">
            <div class="label">Costo final J(θ)</div>
            <div class="value">${final.cost.toFixed(4)}</div>
        </div>
        <div class="final-value-card">
            <div class="label">Ecuación final</div>
            <div class="value" style="font-size: 1rem;">y = ${final.theta_0_real.toFixed(2)} + ${final.theta_1_real.toFixed(2)}x</div>
        </div>
    `;
}

playBtn.addEventListener('click', startAnimation);

pauseBtn.addEventListener('click', stopAnimation);

resetBtn.addEventListener('click', () => {
    stopAnimation();
    currentStep = 0;
    document.getElementById('results').hidden = true;
    renderStep(0);
});

stepBtn.addEventListener('click', () => {
    stopAnimation();
    if (currentStep < history.length - 1) {
        currentStep++;
        renderStep(currentStep);
        if (currentStep === history.length - 1) {
            showResults();
        }
    }
});

speedSlider.addEventListener('input', (e) => {
    speedValue.textContent = `${e.target.value}ms`;
    if (isPlaying) {
        stopAnimation();
        startAnimation();
    }
});