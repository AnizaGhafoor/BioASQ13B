#!/bin/bash


BATCH="Batch04"
# Default values (modify as needed)
DATA_PATH="/data/bioasq13/outputs/${BATCH}/runs_ap/ranx_run_4.json"


OUTPUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_ap/initial_gen"
DIR="/data/bioasq13/phaseB"



MODEL_NAMES=(
"Nemotron"
"OpenBioLLM" 
)  # List of models to iterate over
NUM_ABSTRACTS="3,5,10" # probably should rerank
# NUM_ABSTRACTS="100" # probably should rerank

PROMPT_IDS="1,4,5" 
# PROMPT_IDS="1,4" 
# PROMPT_IDS="5"


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

echo "All models processed!"





# #!/bin/bash


# BATCH="Batch02"
# # Default values (modify as needed)
# DATA_PATH="/data/bioasq13/outputs/${BATCH}/runs_ap/ranx_run_4.json"


# OUTPUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_ap/summaries"
# DIR="/data/bioasq13/phaseB"
# RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_ap/initial_gen"


# MODEL_NAMES=(
# # "OpenBioLLM" 
# "Nemotron"
# )  # List of models to iterate over

# PROMPT_IDS="2" 


# LOGFILE="logs/3_test.log"
# exec > "$LOGFILE" 2>&1






# RUNS=(
# "${RUN_DIR}/OpenBioLLM_abstracts_10_1.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_10_2.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_10_4.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_3_1.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_3_2.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_3_1.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_5_1.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_5_2.json"  
# "${RUN_DIR}/OpenBioLLM_abstracts_5_4.json"  
# )

# # Run the script for each model
# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     echo "Running for model: $MODEL_NAME"
#     python3 ${DIR}/cooking_summaries.py \
#         "${RUNS[@]}" \
#         --model-name "$MODEL_NAME" \
#         --data-path "$DATA_PATH" \
#         --output-dir "${OUTPUT_DIR}" \
#         --prompt-ids "$PROMPT_IDS" \
#         --out-id "4"

#     echo "Completed: $MODEL_NAME"
# done




# RUNS=(
# "${OUTPUT_DIR}/1_Nemotron_summary_3_2.json"
# "${OUTPUT_DIR}/2_Nemotron_summary_3_2.json"
# "${OUTPUT_DIR}/3_Nemotron_summary_3_2.json"
# )

# # Run the script for each model
# for MODEL_NAME in "${MODEL_NAMES[@]}"; do
#     echo "Running for model: $MODEL_NAME"
#     python3 ${DIR}/cooking_summaries.py \
#         "${RUNS[@]}" \
#         --model-name "$MODEL_NAME" \
#         --data-path "$DATA_PATH" \
#         --output-dir "${OUTPUT_DIR}" \
#         --prompt-ids "$PROMPT_IDS" \
#         --out-id "4"

#     echo "Completed: $MODEL_NAME"
# done
# # {out_id}_{model_name}_summary_{len(runs)}_{pid}

# echo "All models processed!"
