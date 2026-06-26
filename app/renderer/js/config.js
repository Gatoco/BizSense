const API = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch(`${API}/config`);
        const config = await res.json();

        document.getElementById('default-alpha').value = config.default_alpha;
        document.getElementById('default-iterations').value = config.default_iterations;
        document.getElementById('theme').value = config.theme;
        document.getElementById('language').value = config.language;
        document.getElementById('ai-provider').value = config.ai_provider;
        document.getElementById('ai-endpoint').value = config.ai_endpoint;

        await loadModels(config.ai_model);
    } catch (err) {
        console.error('Error loading config:', err);
    }

    document.getElementById('ai-provider').addEventListener('change', () => loadModels(''));

    document.getElementById('save-btn').addEventListener('click', async () => {
        const config = {
            default_alpha: parseFloat(document.getElementById('default-alpha').value),
            default_iterations: parseInt(document.getElementById('default-iterations').value),
            theme: document.getElementById('theme').value,
            language: document.getElementById('language').value,
            ai_provider: document.getElementById('ai-provider').value,
            ai_model: document.getElementById('ai-model').value,
            ai_endpoint: document.getElementById('ai-endpoint').value
        };

        try {
            await fetch(`${API}/config`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(config)
            });

            const status = document.getElementById('save-status');
            status.classList.remove('hidden');
            setTimeout(() => status.classList.add('hidden'), 2000);
        } catch (err) {
            alert('Error al guardar configuracion.');
            console.error(err);
        }
    });
});

async function loadModels(selectedModel) {
    const select = document.getElementById('ai-model');
    select.innerHTML = '<option value="">Auto (usar modelo activo)</option>';

    try {
        const res = await fetch(`${API}/ai/models`);
        const data = await res.json();

        if (data.models && data.models.length > 0) {
            data.models.forEach(m => {
                const opt = new Option(m, m);
                if (m === selectedModel) opt.selected = true;
                select.add(opt);
            });
        }

        if (selectedModel && selectedModel !== 'qwen2.5:1.5b' && !data.models.includes(selectedModel)) {
            const opt = new Option(selectedModel + ' (no disponible)', selectedModel);
            select.add(opt);
            select.value = selectedModel;
        }
    } catch (err) {
        console.error('Error loading models:', err);
        const opt = new Option('No se pudieron cargar modelos', '');
        select.add(opt);
    }
}