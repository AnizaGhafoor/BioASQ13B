#!/bin/bash

source /data/bioasq13/venv/bin/activate

LOGFILE="logs/6_Bioasq.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1

BATCH="Batch04"
DIR="/data/bioasq13/phaseA-reranker"
RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_a/"
GS_DIR="/data/bioasq13/outputs/${BATCH}/BioASQ-task13bPhaseA-testset4"


python3  "${DIR}/bioasq_format_converter.py"  "${RUN_DIR}/2025.json"  "${GS_DIR}"

python3  "${DIR}/bioasq_format_converter.py"  "${RUN_DIR}/ranx_run0.json"  "${GS_DIR}"
python3  "${DIR}/bioasq_format_converter.py"  "${RUN_DIR}/ranx_run1.json"  "${GS_DIR}"
python3  "${DIR}/bioasq_format_converter.py"  "${RUN_DIR}/ranx_run2.json"  "${GS_DIR}"
python3  "${DIR}/bioasq_format_converter.py"  "${RUN_DIR}/ranx_run3.json"  "${GS_DIR}"
python3  "${DIR}/bioasq_format_converter.py"  "${RUN_DIR}/ranx_run4.json"  "${GS_DIR}"

zip -r "/data/bioasq13/outputs/${BATCH}/runs_a/${BATCH}_runs_a.zip"  "/data/bioasq13/outputs/${BATCH}/runs_a/runs_bioasq_format"
