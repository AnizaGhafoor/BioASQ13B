

source /data/bioasq13/venv/bin/activate

BATCH=Batch04
DIR="/data/bioasq13/phaseB"



TEST_SET="/data/bioasq13/outputs/${BATCH}/BioASQ-task13bPhaseB-testset4"


RUN_DIR="/data/bioasq13/outputs/${BATCH}/runs_b/initial_gen"
SUMM_DIR="/data/bioasq13/outputs/${BATCH}/runs_b/summaries"
OUT_DIR="/data/bioasq13/outputs/${BATCH}/runs_b/runs_bioasq_format"



LOGFILE="logs/4_convert.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1


python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${RUN_DIR}/Nemotron_snippets_100_5.json" "${OUT_DIR}/run0.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
echo "0"
python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${SUMM_DIR}/1_Nemotron_summary_6_3.json" "${OUT_DIR}/run1.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
echo "1"
python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${RUN_DIR}/bioasq_b4_pB_E1.json" "${OUT_DIR}/run2.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
# python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${SUMM_DIR}/2_Nemotron_summary_8_3.json" "${OUT_DIR}/run2.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
echo "2"
python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${RUN_DIR}/bioasq_b4_pB_E3.json" "${OUT_DIR}/run3.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
# python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${SUMM_DIR}/3_Nemotron_summary_8_3.json" "${OUT_DIR}/run3.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
echo "3"
python3 $DIR/bioasq_format_converter.py  "${TEST_SET}" "${SUMM_DIR}/4_Nemotron_summary_4_3.json" "${OUT_DIR}/run4.json" "${SUMM_DIR}/4_Nemotron_summary_4_3.json"
echo "4"

zip -r "/data/bioasq13/outputs/${BATCH}/runs_b/${BATCH}_runs_b.zip"  ${OUT_DIR}



# runs:

# #Nemotron_5_3.json - i think this is the strongest in isolation

# # RUN 1
# #OpenBioLLM_5_3.json - i think this is the strongest in isolation

# # RUN 2
# # ALL @ 3
# data_files = [
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/Nemotron_3_1.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/Nemotron_3_2.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/Nemotron_3_3.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/OpenBioLLM_3_1.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/OpenBioLLM_3_2.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/OpenBioLLM_3_3.json",
# ]


# # RUN 3
# # ALL @ 5
# data_files = [
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/Nemotron_5_1.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/Nemotron_5_2.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/Nemotron_5_3.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/OpenBioLLM_5_1.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/OpenBioLLM_5_2.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/initial_gen/OpenBioLLM_5_3.json",
# ]

# # Run 4
# # ALL or at least two ensemble of all?
# data_files = [
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/summaries/Nemotron_summary_B2.json",
#     "/data/bioasq13/phaseA-reranker/Batch01/runs_a_p/summaries/Nemotron_summary_B3.json",
# ]