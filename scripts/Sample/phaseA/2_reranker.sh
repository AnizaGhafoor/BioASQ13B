source /data/bioasq13/venv/bin/activate


LOGFILE="logs/3_reranker2.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1



BATCH="Batch04"
DIR="/data/bioasq13/phaseA-reranker"
BASELINE="/data/bioasq13/outputs/${BATCH}/bm25_0.4_0.3.jsonl"


AT_VALUE=100


# 2025 models

declare -a CHECKPOINTS=(
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-8904"
    "michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-4452"
    "microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-4461"
    "michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-17844"
    "michiyasunaga-BioLinkBERT-large-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-8904"
    "microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
    "microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-100-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-8904"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-4452"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-4452"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-2226"
    "michiyasunaga-BioLinkBERT-large-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
    "michiyasunaga-BioLinkBERT-base-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-8904"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E4-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-17844"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue/checkpoint-4452"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sbasicv2-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupTrue/checkpoint-4452"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
    "microsoft-BiomedNLP-BiomedBERT-large-uncased-abstract-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-6678"
    "michiyasunaga-BioLinkBERT-large-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-4461"
    "michiyasunaga-BioLinkBERT-base-42-E1-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse/checkpoint-2226"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-4452"
    "microsoft-BiomedNLP-BiomedBERT-base-uncased-abstract-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
    "michiyasunaga-BioLinkBERT-base-100-E2-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse/checkpoint-8922"
    "michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-quality_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse/checkpoint-6678"
)

# 2:45
# 
OUT="/data/bioasq13/outputs/${BATCH}/runs_a/2025"
for checkpoint in "${CHECKPOINTS[@]}"; do
    python3 "${DIR}/hf_rerank_bert_v2.py" \
            --checkpoint "${DIR}/trained_models_b02/${checkpoint}" \
            --baseline_path "$BASELINE" \
            --at "$AT_VALUE" \
            --path_to_save "$OUT"
done


AT_VALUE=100
declare -a REVISIONS=(
    # Biolink Base (FULL-old data)
    # pairwise full
    "michiyasunaga-BioLinkBERT-base-42-E1-Sexponential-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-7241"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-14482"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sexponential-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-14482"
    "michiyasunaga-BioLinkBERT-base-42-E3-Sexponential-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-21723"
    "michiyasunaga-BioLinkBERT-base-42-E5-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-36205"
    # large full old pairwise
    "michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-21723"
    "michiyasunaga-BioLinkBERT-large-42-E5-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-18070"

    # pubmed bert
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-109560-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E5-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-36205"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-76366-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140"

    "microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-42-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140"
# )



# declare -a REVISIONS=(
    # pairwise val
    "michiyasunaga-BioLinkBERT-base-42-E10-Sexponential-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-69380"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-checkpoint-13876"
    "michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupTrue-checkpoint-13876"
    "michiyasunaga-BioLinkBERT-base-42-E3-Sexponential-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-checkpoint-20814"

    # pointiwse val
    "michiyasunaga-BioLinkBERT-base-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-6926"
    "michiyasunaga-BioLinkBERT-large-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSFalse-warmupFalse-checkpoint-7228"

    # Biolink large
    "michiyasunaga-BioLinkBERT-large-109560-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140"
    "michiyasunaga-BioLinkBERT-large-109560-E5-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-18070"
    "michiyasunaga-BioLinkBERT-large-42-E10-Sbasic-SPbasic-full-old_data-ls0.0-checkpoint-36140"
    "michiyasunaga-BioLinkBERT-large-76366-E10-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-36140"
    "michiyasunaga-BioLinkBERT-large-finetune-checkpoint-17945"
    # Pointwise
    "michiyasunaga-BioLinkBERT-large-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-7228"
    "michiyasunaga-BioLinkBERT-large-42-E4-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpointwise-ExPOSTrue-warmupFalse-checkpoint-14456"

    # val pairwise
    "michiyasunaga-BioLinkBERT-large-42-E2-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-checkpoint-6926"
    "michiyasunaga-BioLinkBERT-large-42-E3-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-checkpoint-20814"
    "michiyasunaga-BioLinkBERT-large-42-E5-Sbasic-SPbasic-val-old_data-CBFalse-KN1-GA1-TRpairwise-checkpoint-17315"

    "microsoft-BiomedNLP-PubMedBERT-large-uncased-abstract-finetune-checkpoint-17945"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-checkpoint-8975"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-109560-E5-Sbasic-SPbasic-full-old_data-CBFalse-checkpoint-18070"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E1-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-7241"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-42-E2-Sbasic-SPbasic-full-old_data-CBFalse-KN1-GA1-TRpairwise-ExPOSTrue-warmupFalse-checkpoint-14482"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-26585-10-full-old_data-checkpoint-36140"
)


OUT="/data/bioasq13/outputs/${BATCH}/runs_a/2024"
for rev in "${REVISIONS[@]}"; do
    python3 "${DIR}/hf_rerank_bert_v2.py" \
            --checkpoint "IEETA/BioASQ-12B" \
            --revision "${rev}" \
            --baseline_path "$BASELINE" \
            --at "$AT_VALUE" \
            --path_to_save "$OUT"
done



# # 2023
declare -a revisions=(
    "V1-checkpoint-25784"
    "V1-checkpoint-32230"
    "V2_synthetic-checkpoint-29268-finetune-checkpoint-9174"
    "V2_synthetic-checkpoint-32490-finetune-checkpoint-10722"
    "V2_synthetic-checkpoint-32490-finetune-checkpoint-11475"
    "V2_synthetic-checkpoint-32490-finetune-checkpoint-11475"
    "V2_synthetic-checkpoint-32490-finetune-checkpoint-12236"
    "V2_synthetic-checkpoint-32490-finetune-checkpoint-12236"
    "V2_synthetic-checkpoint-32490-finetune-checkpoint-5397"
    "V6-checkpoint-24472"
    "V6-checkpoint-24472"
    "V6-checkpoint-27531"
    "V6-checkpoint-27531"
    "V6-checkpoint-30580"
    "V6-checkpoint-30580"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-checkpoint-59000"
    "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-syntotrue-v2-checkpoint-62500"
    "pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475"
    "pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475"
    "pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710"
)

OUT="/data/bioasq13/outputs/${BATCH}/runs_a/2023"
for rev in "${revisions[@]}"; do
    python3 "${DIR}/hf_rerank_bert_v2.py" \
            --checkpoint "T-Almeida/BioASQ-11B" \
            --revision "$rev" \
            --baseline_path "$BASELINE" \
            --at "$AT_VALUE" \
            --path_to_save "$OUT"
done


# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V1-checkpoint-25784" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V1-checkpoint-32230" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-29268-finetune-checkpoint-9174" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-32490-finetune-checkpoint-10722" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-32490-finetune-checkpoint-11475" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-32490-finetune-checkpoint-11475" --baseline_path "$BASELINE" --at 500 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-32490-finetune-checkpoint-12236" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-32490-finetune-checkpoint-12236" --baseline_path "$BASELINE" --at 500 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V2_synthetic-checkpoint-32490-finetune-checkpoint-5397" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V6-checkpoint-24472" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V6-checkpoint-24472" --baseline_path "$BASELINE" --at 1000 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V6-checkpoint-27531" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V6-checkpoint-27531" --baseline_path "$BASELINE" --at 1000 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V6-checkpoint-30580" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "V6-checkpoint-30580" --baseline_path "$BASELINE" --at 1000 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-checkpoint-59000" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "microsoft-BiomedNLP-PubMedBERT-base-uncased-abstract-finetune-0.05-syntotrue-v2-checkpoint-62500" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-11475" --baseline_path "$BASELINE" --at 500 --path_to_save "${OUT}"
# # python3 "${DIR}/hf_rerank_bert_v2.py" --checkpoint "T-Almeida/BioASQ-11B" --revision "pairwise_V1_synthetic-checkpoint-62400-finetune-checkpoint-7710" --baseline_path "$BASELINE" --at 100 --path_to_save "${OUT}"
