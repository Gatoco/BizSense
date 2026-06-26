const API = 'http://localhost:5000';

document.addEventListener('DOMContentLoaded', async () => {
    try {
        const res = await fetch(`${API}/config`);
        const config = await res.json();

        document.getElementById('default-alpha').value = config.default_alpha;
        document.getElementById('default-iterations').value = config.default_iterations;
        document.getElementById('theme').value = config.theme;
        document.getElementById('language').value = config.language;
    } catch (err) {
        console.error('Error loading config:', err);
    }

    document.getElementById('save-btn').addEventListener('click', async () => {
        const config = {
            default_alpha: parseFloat(document.getElementById('default-alpha').value),
            default_iterations: parseInt(document.getElementById('default-iterations').value),
            theme: document.getElementById('theme').value,
            language: document.getElementById('language').value
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