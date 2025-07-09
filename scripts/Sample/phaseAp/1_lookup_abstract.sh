

source /data/bioasq13/venv/bin/activate

BATCH=Batch04
DIR="/data/bioasq13/phaseB/"

PHASEA_DIR="/data/bioasq13/outputs/${BATCH}"
BIOASQ_DIR="/data/bioasq13/outputs/${BATCH}/runs_a/runs_bioasq_format"
OUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_ap"



LOGFILE="logs/1_abstract_lookup.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1


python3 "$DIR/lookup_abstract_Ap.py"  "${BIOASQ_DIR}/ranx_run0.json" "${PHASEA_DIR}/bm25_0.4_0.3.jsonl" "${OUT_DIR}/ranx_run_0.json"
python3 "$DIR/lookup_abstract_Ap.py"  "${BIOASQ_DIR}/ranx_run1.json" "${PHASEA_DIR}/bm25_0.4_0.3.jsonl" "${OUT_DIR}/ranx_run_1.json"
python3 "$DIR/lookup_abstract_Ap.py"  "${BIOASQ_DIR}/ranx_run2.json" "${PHASEA_DIR}/bm25_0.4_0.3.jsonl" "${OUT_DIR}/ranx_run_2.json"
python3 "$DIR/lookup_abstract_Ap.py"  "${BIOASQ_DIR}/ranx_run3.json" "${PHASEA_DIR}/bm25_0.4_0.3.jsonl" "${OUT_DIR}/ranx_run_3.json"
python3 "$DIR/lookup_abstract_Ap.py"  "${BIOASQ_DIR}/ranx_run4.json" "${PHASEA_DIR}/bm25_0.4_0.3.jsonl" "${OUT_DIR}/ranx_run_4.json"

