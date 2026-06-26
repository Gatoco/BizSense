#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Iniciando BizSense..."
echo ""

if [ ! -d ".venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv .venv
fi

echo "Activando entorno virtual..."
source .venv/bin/activate

echo "Instalando dependencias Python..."
pip install -q -r requirements.txt

echo "Instalando dependencias Node..."
npm install --silent

echo ""
echo "Proveedor de IA: deteccion automatica al iniciar"
echo "  - LM Studio: detectado si esta abierto (puerto 1234)"
echo "  - Ollama: detectado si esta corriendo (puerto 11434)"
echo "  - El modelo activo se usara automaticamente en Resultados"
echo ""

echo "Iniciando aplicacion Electron (backend se inicia automaticamente)..."
echo ""
echo "==================================================="
echo "  BizSense"
echo "  Backend: http://localhost:5000"
echo "  Logs backend: /tmp/bizsense-backend.log"
echo "  Logs electron: /tmp/bizsense-electron.log"
echo "==================================================="
echo ""

# Electron main.js se encarga de iniciar y detener el backend
# Fontconfig warnings se redirigen al log de electron
npm start 2>/tmp/bizsense-electron.log

echo ""
echo "BizSense cerrado"