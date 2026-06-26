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
echo "Iniciando backend (FastAPI)..."
.venv/bin/python -m ml.main > /tmp/bizsense-backend.log 2>&1 &
BACKEND_PID=$!

echo "Esperando a que el backend este listo..."
for i in {1..10}; do
    if curl -s http://localhost:5000/docs > /dev/null 2>&1; then
        echo "Backend listo en http://localhost:5000"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "Error: Backend no respondio despues de 10 intentos"
        echo "Logs del backend:"
        cat /tmp/bizsense-backend.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "Iniciando aplicacion Electron..."
echo ""
echo "==================================================="
echo "  BizSense esta corriendo"
echo "  Backend: http://localhost:5000"
echo "  Logs: /tmp/bizsense-backend.log"
echo "==================================================="
echo ""

npm start

echo ""
echo "Deteniendo backend..."
kill $BACKEND_PID 2>/dev/null || true

echo "BizSense cerrado"