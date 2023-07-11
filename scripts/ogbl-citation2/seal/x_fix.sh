dataset=ogbl-citation2
model=SEAL

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt

bash scripts/ogbl-citation2/seal/seal_train.sh --model_type $model --dataset $dataset --suffix $suffix \
    --single_gpu 0 \
    --num_hops 1 \
    --use_feature \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --dynamic_train \
    --dynamic_val \
    --dynamic_test \
    --train_percent 2 \
    --val_percent 1 \
    --test_percent 1 \
    --runs 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt

bash scripts/ogbl-citation2/seal/seal_train.sh --model_type $model --dataset $dataset --suffix $suffix \
    --single_gpu 1 \
    --num_hops 1 \
    --use_feature \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --dynamic_train \
    --dynamic_val \
    --dynamic_test \
    --train_percent 2 \
    --val_percent 1 \
    --test_percent 1 \
    --runs 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt

bash scripts/ogbl-citation2/seal/seal_train.sh --model_type $model --dataset $dataset --suffix $suffix \
    --single_gpu 2 \
    --num_hops 1 \
    --use_feature \
    --use_edge_weight \
    --eval_steps 1 \
    --epochs 10 \
    --dynamic_train \
    --dynamic_val \
    --dynamic_test \
    --train_percent 2 \
    --val_percent 1 \
    --test_percent 1 \
    --runs 1 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
