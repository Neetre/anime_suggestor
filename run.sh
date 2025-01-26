#!/bin/bash

cd /site/anime_suggestor


setup_venv() {
    echo "Setting up virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install -r requirements.txt
}

if [ ! -d ".venv" ]; then
    setup_venv
else
    source .venv/bin/activate
fi

cd bin

echo "Starting Python API..."
uvicorn api:app --host 0.0.0.0 --port 8006 &

sleep 3

echo "Starting Frontend..."
streamlit run ./interface.py --server.port=8007 --server.address=0.0.0.0

wait