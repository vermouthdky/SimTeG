dataset=ogbn-arxiv
model_type=GCN
lm_model_type=all-roberta-large-v1
suffix=X_${lm_model_type}_peft

bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt

bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --single_gpu 0 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
