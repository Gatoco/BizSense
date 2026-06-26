const API = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', () => {
    loadDatasets();

    document.getElementById('csv-input').addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API}/datasets`, { method: 'POST', body: formData });
            const data = await res.json();

            document.getElementById('upload-result').innerHTML = `
                <p class="text-primary">Dataset "${data.name}" cargado: ${data.rows} filas, ${data.columns.length} columnas</p>
            `;
            document.getElementById('upload-result').classList.remove('hidden');

            loadDatasets();
        } catch (err) {
            alert('Error al subir archivo.');
            console.error(err);
        }
    });
});

async function loadDatasets() {
    try {
        const res = await fetch(`${API}/datasets`);
        const datasets = await res.json();

        const container = document.getElementById('datasets-list');

        if (datasets.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No hay datasets cargados</p></div>';
            return;
        }

        let html = '<div class="table-wrapper"><table><thead><tr>';
        html += '<th>ID</th><th>Nombre</th><th>Columnas</th><th>Filas</th><th>Fecha</th><th>Acciones</th>';
        html += '</tr></thead><tbody>';

        datasets.forEach(ds => {
            html += `<tr>
                <td>${ds.id}</td>
                <td>${ds.name}</td>
                <td>${ds.columns.join(', ')}</td>
                <td>${ds.rows}</td>
                <td>${ds.created_at || '-'}</td>
                <td>
                    <button class="btn btn-sm" onclick="showPreview(${ds.id})">Ver</button>
                    <button class="btn btn-sm btn-danger" onclick="deleteDataset(${ds.id})">Eliminar</button>
                </td>
            </tr>`;
        });

        html += '</tbody></table></div>';
        container.innerHTML = html;
    } catch (err) {
        console.error('Error loading datasets:', err);
    }
}

async function showPreview(id) {
    try {
        const res = await fetch(`${API}/datasets/${id}`);
        const data = await res.json();

        document.getElementById('preview-name').textContent = data.name;
        document.getElementById('preview-card').classList.remove('hidden');

        let html = '<div class="table-wrapper"><table><thead><tr>';
        data.columns.forEach(col => html += `<th>${col}</th>`);
        html += '</tr></thead><tbody>';

        data.preview.forEach(row => {
            html += '<tr>';
            data.columns.forEach(col => {
                const val = row[col];
                html += `<td>${val !== null && val !== undefined ? val : '-'}</td>`;
            });
            html += '</tr>';
        });

        html += '</tbody></table></div>';
        document.getElementById('preview-table').innerHTML = html;
    } catch (err) {
        console.error('Error loading preview:', err);
    }
}

async function deleteDataset(id) {
    if (!confirm('Eliminar este dataset?')) return;

    try {
        await fetch(`${API}/datasets/${id}`, { method: 'DELETE' });
        loadDatasets();
        document.getElementById('preview-card').classList.add('hidden');
    } catch (err) {
        alert('Error al eliminar.');
        console.error(err);
    }
}