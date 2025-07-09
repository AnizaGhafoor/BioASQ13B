#!/bin/bash


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


LOGFILE="logs/3_summaries.log"
exec > "$LOGFILE" 2>&1




# RUN 0
# Best nemotron

# RUN 1
# Best Customm

# RUN 2
# PROMPT 5
RUNS=(
"${RUN_DIR}/Nemotron_abstracts_10_5.json"
"${RUN_DIR}/Nemotron_abstracts_3_5.json"  
"${RUN_DIR}/Nemotron_abstracts_5_5.json"  
)  


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running for model: $MODEL_NAME"
    python3 ${DIR}/summaries.py \
        "${RUNS[@]}" \
        --model-name "$MODEL_NAME" \
        --data-path "$DATA_PATH" \
        --output-dir "${OUTPUT_DIR}" \
        --prompt-ids "$PROMPT_IDS" \
        --out-id "2"

    echo "Completed: $MODEL_NAME"
done

# RUN 3
# RUN2 different input order 
RUNS=(
"${RUN_DIR}/Nemotron_abstracts_3_5.json"  
"${RUN_DIR}/Nemotron_abstracts_5_5.json"  
"${RUN_DIR}/Nemotron_abstracts_10_5.json"

)  


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running for model: $MODEL_NAME"
    python3 ${DIR}/summaries.py \
        "${RUNS[@]}" \
        --model-name "$MODEL_NAME" \
        --data-path "$DATA_PATH" \
        --output-dir "${OUTPUT_DIR}" \
        --prompt-ids "$PROMPT_IDS" \
        --out-id "3"

    echo "Completed: $MODEL_NAME"
done

# RUN 4
#large natcj
RUNS=(
"${RUN_DIR}/Nemotron_abstracts_10_1.json"  
"${RUN_DIR}/Nemotron_abstracts_10_5.json"  
"${RUN_DIR}/Nemotron_abstracts_10_4.json"  
"${RUN_DIR}/Nemotron_abstracts_3_1.json"   
"${RUN_DIR}/Nemotron_abstracts_3_5.json"
"${RUN_DIR}/Nemotron_abstracts_3_4.json"  
"${RUN_DIR}/Nemotron_abstracts_5_1.json"  
"${RUN_DIR}/Nemotron_abstracts_5_5.json"  
"${RUN_DIR}/Nemotron_abstracts_5_4.json"
)  


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running for model: $MODEL_NAME"
    python3 ${DIR}/summaries.py \
        "${RUNS[@]}" \
        --model-name "$MODEL_NAME" \
        --data-path "$DATA_PATH" \
        --output-dir "${OUTPUT_DIR}" \
        --prompt-ids "$PROMPT_IDS" \
        --out-id "4"

    echo "Completed: $MODEL_NAME"
done



# RUN 4
# MERGE of 2




echo "All models processed!"
