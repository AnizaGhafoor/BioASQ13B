source /data/bioasq13/venv/bin/activate

DIR="/data/bioasq13/phaseA-reranker"

BATCH="Batch04"

TESTSET="/data/bioasq13/outputs/${BATCH}/BioASQ-task13bPhaseA-testset4"

LOGFILE="logs/4_dprf.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1


# note this batch we reduced the dprf stuff.

cd $DIR

RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_a/2025"
MODEL_DIR="/data/bioasq13/phaseA-reranker/trained_models_b02"


MODELS=(
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-4452"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-2226"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-4452"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue/checkpoint-4452"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-8904"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-17844"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-8904"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-4461"
"${MODEL_DIR}/michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-6678"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-4452"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-2226"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-4452"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue/checkpoint-4452"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-8904"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-17844"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-8904"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-4461"
"${MODEL_DIR}/microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-6678"
)

RUNS=(
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-2226_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue_checkpoint-4452_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-17844_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-4461_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-6678_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-2226_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue_checkpoint-4452_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-17844_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-4461_bm25_0.4_0.3_100.json"
"${RUN_DIR}/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-6678_bm25_0.4_0.3_100_relevance.json"
)


python pseudo_relevance_feedback.py $TESTSET --ranx_runs "${RUNS[@]}" --model_checkpoints "${MODELS[@]}" 

deactivate