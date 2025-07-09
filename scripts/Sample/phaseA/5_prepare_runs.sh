#!/bin/bash
source /data/bioasq13/venv/bin/activate
BATCH="Batch04"
DIR="/data/bioasq13/phaseA-reranker"

LOGFILE="logs/5_prepare_runs.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1


RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_a/"



# Run 2023

FILES=(
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-25784_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-25784_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-32230_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V1-checkpoint-32230_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-29268-finetune-checkpoint-9174_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-29268-finetune-checkpoint-9174_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-10722_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-10722_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-11475_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-11475_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-12236_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-12236_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-5397_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V2_synthetic-checkpoint-32490-finetune-checkpoint-5397_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-24472_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-24472_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-27531_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-27531_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-30580_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_V6-checkpoint-30580_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-checkpoint-59000_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-checkpoint-59000_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-syntotrue-v2-checkpoint-62500_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-syntotrue-v2-checkpoint-62500_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2023/ranx_T-Almeida_BioASQ-11B_pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710_bm25_0.4_0.3_100.json"

)
python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/2023.json" --method rrf
echo "-----------------"



# RUN 2024
#  -----------------------------------
# 2024 models
FILES=(
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E1-Sexponential-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-7241_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E10-Sexponential-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-69380_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-14482_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-checkpoint-13876_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupTrue-checkpoint-13876_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6926_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-14482_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E3-Sexponential-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-21723_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E3-Sexponential-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-checkpoint-20814_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-base-42-E5-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-36205_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-109560-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-109560-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-109560-E5-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-18070_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-109560-E5-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-18070_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E10-Sbasic-SPbasic-full-old_data-ls0.0-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E10-Sbasic-SPbasic-full-old_data-ls0.0-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-7228_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-7228_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-checkpoint-6926_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-21723_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-checkpoint-20814_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E4-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-14456_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E5-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-18070_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-42-E5-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-checkpoint-17315_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-76366-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-76366-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-finetune-checkpoint-17945_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_michiyasunaga-BioLinkBERT-large-finetune-checkpoint-17945_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-109560-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-109560-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-109560-E5-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-18070_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-109560-E5-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-18070_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-26585-10-full-old_data-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-26585-10-full-old_data-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-7241_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-14482_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E5-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-36205_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-76366-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-76366-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-checkpoint-8975_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-checkpoint-8975_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-42-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-42-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140_bm25_0.4_0.3_100_relevance.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-finetune-checkpoint-17945_bm25_0.4_0.3_100.json"
"${RUN_DIR}/2024/ranx_IEETA_BioASQ-12B_microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-finetune-checkpoint-17945_bm25_0.4_0.3_100_relevance.json"   
)
python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/2024.json" --method rrf
echo "-----------------"



# 2025
FILES=(
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-2226_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_relevance.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue_checkpoint-4452_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-17844_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_relevance.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-4461_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-6678_bm25_0.4_0.3_100_relevance.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-2226_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_relevance.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue_checkpoint-4452_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-17844_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_relevance.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-4461_bm25_0.4_0.3_100.json"
    "${RUN_DIR}/2025/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-6678_bm25_0.4_0.3_100_relevance.json"
)

python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/ranx_run0.json" --method rrf



# python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/ranx_run0.json" --method rrf
echo "-----------------"

#  -----------------------------------
#DPRF
FILES=(
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-2226_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_relevance_dprf_relevance.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue_checkpoint-4452_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-17844_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_relevance_dprf_relevance.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-4461_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-6678_bm25_0.4_0.3_100_relevance_dprf_relevance.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-2226_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-4452_bm25_0.4_0.3_100_relevance_dprf_relevance.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue_checkpoint-4452_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-17844_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-8922_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-8904_bm25_0.4_0.3_100_relevance_dprf_relevance.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse_checkpoint-4461_bm25_0.4_0.3_100_dprf.json"
"${RUN_DIR}/2025_dprf/ranx_microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse_checkpoint-6678_bm25_0.4_0.3_100_relevance_dprf_relevance.json"

   )

python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/ranx_run1.json" --method rrf
echo "-----------------"


# RUN 0 (baseline)
# 2025

# RUN 1
# 2025_dprf



# RUN 2
#  -----------------------------------
# all models no dprf

FILES=(
    "${RUN_DIR}/2023.json"
    "${RUN_DIR}/2024.json"
    "${RUN_DIR}/ranx_run0.json"
    )


python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/ranx_run2.json" --method rrf
echo "-----------------"

# RUN 4
#  -----------------------------------
# dprf 2025 and  2024 (test if next is actually better)

FILES=(
    "${RUN_DIR}/2024.json"
    "${RUN_DIR}/ranx_run1.json"
    )

python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/ranx_run3.json" --method rrf
echo "-----------------"



# RUN 4
#  -----------------------------------
# dprf 2025 and ALL 

FILES=(
    "${RUN_DIR}/2023.json"
    "${RUN_DIR}/2024.json"
    "${RUN_DIR}/ranx_run1.json"
    )

python3 "${DIR}/fusion.py" "${FILES[@]}" --out "${RUN_DIR}/ranx_run4.json" --method rrf
echo "-----------------"


