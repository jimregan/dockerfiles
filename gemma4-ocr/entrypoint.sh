#!/bin/bash
ollama serve &
until curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; do
    sleep 2
done
python3 /app/gemma4_ocr.py --input-root "$DATA_DIR" --output-root "$OUTPUT_DIR"
