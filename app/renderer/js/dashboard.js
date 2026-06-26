const API = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch(`${API}/stats`);
        const stats = await res.json();

        document.getElementById('stat-datasets').textContent = stats.datasets_count;
        document.getElementById('stat-trainings').textContent = stats.trainings_count;
        document.getElementById('stat-models').textContent = stats.models_implemented;

        if (stats.last_training) {
            const lt = stats.last_training;
            document.getElementById('last-training-content').innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;">
                    <div>
                        <p style="margin-bottom: 8px;"><span class="text-muted">Modelo:</span> <span class="badge badge-info">${lt.model_type}</span></p>
                        <p style="margin-bottom: 8px;"><span class="text-muted">Dataset:</span> ${lt.dataset_name}</p>
                        <p style="margin-bottom: 8px;"><span class="text-muted">Ecuacion:</span> <span class="text-primary">${lt.equation}</span></p>
                        <p style="margin-bottom: 8px;"><span class="text-muted">Costo final:</span> ${lt.final_cost.toFixed(4)}</p>
                        <p class="text-muted" style="font-size: 0.8rem;">${lt.created_at}</p>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 8px;">
                        <a href="graphics.html" class="btn btn-primary btn-sm">Ir a Funcion Grafica</a>
                        <a href="models.html" class="btn btn-sm">Ver modelos</a>
                    </div>
                </div>
            `;
        }
    } catch (err) {
        console.error('Error loading stats:', err);
    }
});