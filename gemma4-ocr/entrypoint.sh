#!/bin/bash
ollama serve &
sleep 5
python3 /app/gemma4_ocr.py --input-root "$DATA_DIR" --output-root "$OUTPUT_DIR"
