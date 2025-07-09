#!/bin/bash


BATCH="Batch04"
# Default values (modify as needed)
DATA_PATH="/data/bioasq13/outputs/${BATCH}/runs_b/phase_b_abstracts.json"


OUTPUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_b/initial_gen"
DIR="/data/bioasq13/phaseB"


MODEL_NAMES=(
"OpenBioLLM" 
"Nemotron"
)  # List of models to iterate over
NUM_ABSTRACTS="3,5,10" # probably should rerank

PROMPT_IDS="1,4,5" 

LOGFILE="logs/2_initial_gen.log"
exec > "$LOGFILE" 2>&1


# Run the script for each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running for model: $MODEL_NAME"
    python3 ${DIR}/initial_generation.py \
        --model-name "$MODEL_NAME" \
        --data-path "$DATA_PATH" \
        --data-type "abstracts" \
        --num-abstracts "$NUM_ABSTRACTS" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-ids "$PROMPT_IDS"
    echo "Completed: $MODEL_NAME"
done


MODEL_NAMES=(
"OpenBioLLM" 
"Nemotron"
)  # List of models to iterate over
NUM_ABSTRACTS="100" # probably should rerank

PROMPT_IDS="1,4,5" 


LOGFILE="logs/2_initial_gen_snippets.log"
exec > "$LOGFILE" 2>&1



# Run the script for each model
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running for model: $MODEL_NAME"
    python3 ${DIR}/initial_generation.py \
        --model-name "$MODEL_NAME" \
        --data-path "$DATA_PATH" \
        --data-type "snippets" \
        --num-abstracts "$NUM_ABSTRACTS" \
        --output-dir "$OUTPUT_DIR" \
        --prompt-ids "$PROMPT_IDS"
    echo "Completed: $MODEL_NAME"
done

echo "All models processed!"
