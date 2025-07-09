source /data/bioasq13/venv/bin/activate


LOGFILE="logs/3_relevance_converter.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1


BATCH="Batch04"
DIR="/data/bioasq13/phaseA-reranker"



for file in /data/bioasq13/outputs/${BATCH}/runs_a/*/*.json; do
    # Check if the file exists
    if [ -e "$file" ]; then
        # Call Python script with the filename as argument
        if [[ "$file" != *_relevance.json && "$file" != *pairwise* ]]; then
            # Call Python script with the filename as argument
            python3 "${DIR}/relevance_converter.py" "$file"
        else
            echo "Skipping $file as it ends with '_relevance.json'."
        fi
    else
        echo "File $file does not exist."
    fi
done


