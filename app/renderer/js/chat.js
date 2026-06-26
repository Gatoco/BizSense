const API = 'http://localhost:5000';
let chatMessages = [];
let isWaiting = false;

document.addEventListener('DOMContentLoaded', async () => {
    await checkAIStatus();
    await loadChatHistory();

    document.getElementById('ai-start-btn').addEventListener('click', startModel);
    document.getElementById('ai-clear-btn').addEventListener('click', clearChat);
    document.getElementById('chat-send-btn').addEventListener('click', sendMessage);
    document.getElementById('chat-input').addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isWaiting) sendMessage();
    });
});

async function checkAIStatus() {
    try {
        const res = await fetch(`${API}/ai/status`);
        const data = await res.json();

        const indicator = document.getElementById('ai-status-indicator');
        const info = document.getElementById('ai-provider-info');
        const startBtn = document.getElementById('ai-start-btn');
        const clearBtn = document.getElementById('ai-clear-btn');
        const input = document.getElementById('chat-input');
        const sendBtn = document.getElementById('chat-send-btn');

        if (data.available) {
            indicator.textContent = `Conectado - ${data.provider}`;
            indicator.className = 'badge badge-success';
            info.textContent = `Modelo: ${data.model}`;
            startBtn.classList.add('hidden');
            clearBtn.classList.remove('hidden');
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
        } else {
            indicator.textContent = 'No disponible';
            indicator.className = 'badge badge-danger';
            info.textContent = `Esperando: ${data.provider || 'Ollama/LM Studio'}`;
            startBtn.classList.remove('hidden');
            clearBtn.classList.add('hidden');
            input.disabled = true;
            sendBtn.disabled = true;
        }
    } catch (err) {
        console.error('Error checking AI status:', err);
    }
}

async function startModel() {
    const btn = document.getElementById('ai-start-btn');
    btn.disabled = true;
    btn.textContent = 'Iniciando...';

    try {
        const res = await fetch(`${API}/ai/start`, { method: 'POST' });
        const data = await res.json();

        if (data.started) {
            await checkAIStatus();
        } else {
            alert('No se pudo iniciar el modelo. Asegurate de tener Ollama instalado.');
        }
    } catch (err) {
        alert('Error al iniciar modelo.');
        console.error(err);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Iniciar modelo';
    }
}

async function loadChatHistory() {
    try {
        const res = await fetch(`${API}/chat/history`);
        const msgs = await res.json();
        chatMessages = msgs.map(m => ({ role: m.role, content: m.content }));

        const container = document.getElementById('chat-messages');
        container.innerHTML = '';

        if (chatMessages.length === 0) {
            container.innerHTML = '<p class="text-muted" style="text-align: center; padding: 20px;">No hay mensajes. Haz una pregunta sobre tus resultados.</p>';
        } else {
            chatMessages.forEach(msg => appendMessage(msg.role, msg.content));
        }
    } catch (err) {
        console.error('Error loading chat history:', err);
    }
}

async function clearChat() {
    if (!confirm('Limpiar historial de chat?')) return;
    try {
        await fetch(`${API}/chat/history`, { method: 'DELETE' });
        chatMessages = [];
        loadChatHistory();
    } catch (err) {
        console.error('Error clearing chat:', err);
    }
}

function appendMessage(role, content) {
    const container = document.getElementById('chat-messages');

    const msgDiv = document.createElement('div');
    msgDiv.style.cssText = `
        margin-bottom: 12px;
        padding: 12px 16px;
        border-radius: 8px;
        max-width: 85%;
        ${role === 'user'
            ? 'margin-left: auto; background: rgba(0, 212, 255, 0.15); border-left: 3px solid var(--primary);'
            : 'background: var(--surface); border-left: 3px solid var(--muted);'
        }
    `;

    const roleLabel = document.createElement('div');
    roleLabel.textContent = role === 'user' ? 'Tu' : 'IA';
    roleLabel.style.cssText = `font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; font-weight: 600;`;

    const contentDiv = document.createElement('div');
    contentDiv.textContent = content;
    contentDiv.style.cssText = 'font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap;';

    msgDiv.appendChild(roleLabel);
    msgDiv.appendChild(contentDiv);
    container.appendChild(msgDiv);

    container.scrollTop = container.scrollHeight;
}

async function sendMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const text = input.value.trim();

    if (!text || isWaiting) return;

    isWaiting = true;
    input.disabled = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '...';

    chatMessages.push({ role: 'user', content: text });
    appendMessage('user', text);
    input.value = '';

    const container = document.getElementById('chat-messages');
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading-msg';
    loadingDiv.style.cssText = 'margin-bottom: 12px; padding: 12px 16px; background: var(--surface); border-radius: 8px; border-left: 3px solid var(--muted); max-width: 85%;';
    loadingDiv.innerHTML = '<div style="font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; font-weight: 600;">IA</div><div style="font-size: 0.9rem; color: var(--muted);">Escribiendo...</div>';
    container.appendChild(loadingDiv);
    container.scrollTop = container.scrollHeight;

    try {
        const res = await fetch(`${API}/ai/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: chatMessages.filter(m => m.role !== 'system'),
                model: null
            })
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let fullResponse = '';

        loadingDiv.remove();

        const responseDiv = document.createElement('div');
        responseDiv.style.cssText = 'margin-bottom: 12px; padding: 12px 16px; background: var(--surface); border-radius: 8px; border-left: 3px solid var(--muted); max-width: 85%;';
        responseDiv.innerHTML = '<div style="font-size: 0.75rem; color: var(--muted); margin-bottom: 4px; font-weight: 600;">IA</div><div id="streaming-text" style="font-size: 0.9rem; line-height: 1.6; white-space: pre-wrap;"></div>';
        container.appendChild(responseDiv);
        const textDiv = responseDiv.querySelector('#streaming-text');

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                try {
                    const obj = JSON.parse(line.slice(6));
                    if (obj.token) {
                        fullResponse = obj.token;
                        textDiv.textContent = fullResponse;
                        container.scrollTop = container.scrollHeight;
                    }
                    if (obj.error) {
                        textDiv.textContent = 'Error: ' + obj.error;
                        textDiv.style.color = 'var(--danger)';
                    }
                } catch (e) {}
            }
        }

        if (fullResponse) {
            chatMessages.push({ role: 'assistant', content: fullResponse });
        }
    } catch (err) {
        console.error('Chat error:', err);
        appendMessage('assistant', 'Error de conexion con el modelo.');
    } finally {
        document.getElementById('loading-msg')?.remove();
        isWaiting = false;
        input.disabled = false;
        sendBtn.disabled = false;
        sendBtn.textContent = 'Enviar';
        input.focus();
    }
}