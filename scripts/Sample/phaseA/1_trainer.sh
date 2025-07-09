source /data/bioasq13/venv/bin/activate

DIR="/data/bioasq13/phaseA-reranker"


LOGFILE="logs/1_trainer.log"
# Redirect stdout (1) and stderr (2) to a file
exec > "$LOGFILE" 2>&1


cd $DIR


#notes prioritis lower epoch base models < 5


#target is around 20 base 
# 5/10 large
#unverified if 2 epochs is ok

# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 1 --sampler basic --pairwise
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 100 --epoch 2 --sampler basic --pairwise
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 4 --sampler basic --pairwise
# #pointwise
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 2 --sampler basic --use_expanded_pos
#wierd combo
#failes
# expaned pos
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 4 --sampler basic --pairwise --use_expanded_pos
# #  broken
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 2 --sampler exponential --pairwise --use_expanded_pos
# #quantituy
# #seed
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 100 --epoch 2 --sampler basic --pairwise --use_expanded_pos

# #change model

# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 1 --sampler basic --pairwise
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 100 --epoch 2 --sampler basic --pairwise
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 4 --sampler basic --pairwise
# #pointwise
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 2 --sampler basic --use_expanded_pos
# #wierd combo
# # expaned pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 4 --sampler basic --pairwise --use_expanded_pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 2 --sampler exponential --pairwise --use_expanded_pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 100 --epoch 2 --sampler basic --pairwise --use_expanded_pos

#large models
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-large False quality --seed 42 --epoch 1 --sampler basic --pairwise --use_expanded_pos
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-large False quality --seed 42 --epoch 3 --sampler basic --use_expanded_pos
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-large False quality --seed 100 --epoch 2 --sampler basic --pairwise --use_expanded_pos
# python hf_bert_trainer.py michiyasunaga/BioLinkBERT-large False quality --seed 100 --epoch 4 --sampler basic --use_expanded_pos



# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract False quality --seed 42 --epoch 1 --sampler basic --pairwise --use_expanded_pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract False quality --seed 42 --epoch 3 --sampler basic --use_expanded_pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract False quality --seed 100 --epoch 2 --sampler basic --pairwise --use_expanded_pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract False quality --seed 100 --epoch 4 --sampler basic --use_expanded_pos



python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 2 --sampler basicv2 --pairwise --use_expanded_pos
python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quantity --seed 42 --epoch 2 --sampler basic  --use_expanded_pos
python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 2 --sampler basicv2 --pairwise --warmup_ratio
python hf_bert_trainer.py michiyasunaga/BioLinkBERT-base False quality --seed 42 --epoch 2 --sampler basicv2 --pairwise --warmup_ratio
python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 42 --epoch 2 --sampler basicv2 --pairwise --use_expanded_pos

#quantituy
python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quantity --seed 42 --epoch 2 --sampler basic  --use_expanded_pos
# python hf_bert_trainer.py microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract False quality --seed 100 --epoch 2 --sampler basic --pairwise

