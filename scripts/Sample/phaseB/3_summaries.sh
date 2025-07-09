#!/bin/bash




BATCH="Batch04"
# Default values (modify as needed)
DATA_PATH="/data/bioasq13/outputs/${BATCH}/runs_b/phase_b_abstracts.json"


OUTPUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_b/summaries"
DIR="/data/bioasq13/phaseB"
RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_b/initial_gen"


MODEL_NAMES=(
# "OpenBioLLM"  #model is ass like half the stuff does not generate valid json
"Nemotron"
)  # List of models to iterate over

PROMPT_IDS="2,3" 


LOGFILE="logs/3_summaries.log"
exec > "$LOGFILE" 2>&1


# "${RUN_DIR}/Nemotron_abstracts_10_1.json"  
# "${RUN_DIR}/Nemotron_abstracts_10_4.json"  
# "${RUN_DIR}/Nemotron_abstracts_10_5.json"  
# "${RUN_DIR}/Nemotron_abstracts_3_1.json"   
# "${RUN_DIR}/Nemotron_abstracts_3_4.json"
# "${RUN_DIR}/Nemotron_abstracts_3_5.json"  
# "${RUN_DIR}/Nemotron_abstracts_5_1.json"  
# "${RUN_DIR}/Nemotron_abstracts_5_4.json"  
# "${RUN_DIR}/Nemotron_abstracts_5_5.json"
# "${RUN_DIR}/Nemotron_snippets_100_1.json"
# "${RUN_DIR}/Nemotron_snippets_100_4.json"
# "${RUN_DIR}/Nemotron_snippets_100_5.json"


# "${RUN_DIR}/OpenBioLLM_abstracts_10_1.json" 
# "${RUN_DIR}/OpenBioLLM_abstracts_10_4.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_10_5.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_3_1.json"   
# "${RUN_DIR}/OpenBioLLM_abstracts_3_4.json"
# "${RUN_DIR}/OpenBioLLM_abstracts_3_5.json"
# "${RUN_DIR}/OpenBioLLM_abstracts_5_1.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_5_4.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_5_5.json"  
# "${RUN_DIR}/OpenBioLLM_snippets_100_1.json"  
# "${RUN_DIR}/OpenBioLLM_snippets_100_4.json"  
# "${RUN_DIR}/OpenBioLLM_snippets_100_5.json"  





# RUN 0
# Nemotron_snippets_100_5


# All prompt 4
# All promtp 5,
# Nemotron 4,5
 
# RUN 1
# ALL snippets
RUNS=(
"${RUN_DIR}/Nemotron_snippets_100_1.json"
"${RUN_DIR}/Nemotron_snippets_100_4.json"
"${RUN_DIR}/Nemotron_snippets_100_5.json"
"${RUN_DIR}/OpenBioLLM_snippets_100_1.json"  
"${RUN_DIR}/OpenBioLLM_snippets_100_4.json"  
"${RUN_DIR}/OpenBioLLM_snippets_100_5.json"  
)  


for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    echo "Running for model: $MODEL_NAME"
    python3 ${DIR}/summaries.py \
        "${RUNS[@]}" \
        --model-name "$MODEL_NAME" \
        --data-path "$DATA_PATH" \
        --output-dir "${OUTPUT_DIR}" \
        --prompt-ids "$PROMPT_IDS" \
        --out-id "1"

    echo "Completed: $MODEL_NAME"
done

# RUN 2
# All 4

RUNS=(

"${RUN_DIR}/Nemotron_abstracts_10_4.json"  
"${RUN_DIR}/Nemotron_abstracts_3_4.json"
"${RUN_DIR}/Nemotron_abstracts_5_4.json"  
"${RUN_DIR}/Nemotron_snippets_100_4.json"

"${RUN_DIR}/OpenBioLLM_abstracts_10_4.json"  
"${RUN_DIR}/OpenBioLLM_abstracts_5_4.json"  
"${RUN_DIR}/OpenBioLLM_abstracts_3_4.json"
"${RUN_DIR}/OpenBioLLM_snippets_100_4.json"  
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
# All 5
RUNS=(
"${RUN_DIR}/Nemotron_abstracts_10_5.json"  
"${RUN_DIR}/Nemotron_abstracts_3_5.json"
"${RUN_DIR}/Nemotron_abstracts_5_5.json"  
"${RUN_DIR}/Nemotron_snippets_100_5.json"

"${RUN_DIR}/OpenBioLLM_abstracts_10_5.json"  
"${RUN_DIR}/OpenBioLLM_abstracts_5_5.json"  
"${RUN_DIR}/OpenBioLLM_abstracts_3_5.json"
"${RUN_DIR}/OpenBioLLM_snippets_100_5.json"  
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
# Nemotron 5


RUNS=(
"${RUN_DIR}/Nemotron_abstracts_10_5.json"  
"${RUN_DIR}/Nemotron_abstracts_3_5.json"
"${RUN_DIR}/Nemotron_abstracts_5_5.json"  
"${RUN_DIR}/Nemotron_snippets_100_5.json"
)

# Run the script for each model
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
# {out_id}_{model_name}_summary_{len(runs)}_{pid}

echo "All models processed!"
