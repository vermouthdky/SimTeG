dataset=ogbl-citation2
model_type=e5-large
suffix=fix

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --eval_batch_size 500 \
    --mode 'test'

# lm_type=${model_type}
# bert_x_dir=out/${dataset}/${lm_type}/${suffix}/cached_embs/iter_0_x_embs.pt

# for model_type in GAMLP SAGN; do
#     suffix=optuna_on_X_${lm_type}
#     bash scripts/optuna.sh --model_type $model_type --dataset $dataset --suffix $suffix \
#         --pretrained_repo sentence-transformers/${lm_type} \
#         --lm_type $lm_type \
#         --gnn_eval_interval 5 \
#         --gnn_batch_size 10000 \
#         --gnn_eval_batch_size 10000 \
#         --gnn_epochs 100 \
#         --lr_scheduler_type constant \
#         --use_bert_x \
#         --bert_x_dir $bert_x_dir
# done
