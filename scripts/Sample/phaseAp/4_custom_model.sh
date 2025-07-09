#!/bin/bash


source /data/bioasq13/venv-unsloth/bin/activate


BATCH="Batch04"
# Default values (modify as needed)
DATA_PATH="/data/bioasq13/outputs/${BATCH}/runs_ap/ranx_run_4.json"


OUTPUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_ap/summaries"
DIR="/data/bioasq13/phaseB"
RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_ap/initial_gen"


MODEL_NAMES=(
# "OpenBioLLM" 
"Nemotron"
)  # List of models to iterate over

PROMPT_IDS="2" 


LOGFILE="logs/4_custom.log"
exec > "$LOGFILE" 2>&1

python /data/bioasq13/phaseB/infrence_custom.py \
  --model_dir "/data/lmdeploy/gemma-ft/" \
  --output_template "bioasq_b4_run4_"\
  --input_file "/data/bioasq13/outputs/Batch04/runs_ap/ranx_run_4.json"

python /data/bioasq13/phaseB/infrence_custom.py \
  --model_dir "/data/lmdeploy/gemma-ft/" \
  --output_template "bioasq_b4_run3_"\
  --input_file "/data/bioasq13/outputs/Batch04/runs_ap/ranx_run_3.json"

python /data/bioasq13/phaseB/infrence_custom.py \
  --model_dir "/data/lmdeploy/gemma-ft/" \
  --output_template "bioasq_b4_run2_"\
  --input_file "/data/bioasq13/outputs/Batch04/runs_ap/ranx_run_2.json"

python /data/bioasq13/phaseB/infrence_custom.py \
  --model_dir "/data/lmdeploy/gemma-ft/" \
  --output_template "bioasq_b4_run1_"\
  --input_file "/data/bioasq13/outputs/Batch04/runs_ap/ranx_run_1.json"



echo "All models processed!"
