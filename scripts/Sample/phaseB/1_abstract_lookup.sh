

source /data/bioasq13/venv/bin/activate

BATCH=Batch04
DIR="/data/bioasq13/phaseB"

PHASEA_DIR="/data/bioasq13/outputs/${BATCH}"
OUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_b"


TEST_SET="/data/bioasq13/outputs/${BATCH}/BioASQ-task13bPhaseB-testset4"

LOGFILE="logs/1_abstract_lookup.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1



python3 "$DIR/lookup_abstract_B.py"  "${TEST_SET}" "${PHASEA_DIR}/runs_b/phase_b_abstracts.json"

