dataset=ogbl-citation2
model_type=GraphSAGE

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_layer_3
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 0 \
    --lm_type $lm_model_type \
    --gnn_batch_size 512 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 150 \
    --gnn_dropout 0.2 \
    --gnn_lr 0.01 \
    --gnn_num_layers 3 \
    --gnn_weight_decay 2e-6 \
    --gnn_eval_warmup 10 \
    --gnn_eval_interval 5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}_layer_3
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 1 \
    --lm_type $lm_model_type \
    --gnn_batch_size 512 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 150 \
    --gnn_dropout 0.2 \
    --gnn_lr 0.01 \
    --gnn_num_layers 3 \
    --gnn_weight_decay 2e-6 \
    --gnn_eval_warmup 10 \
    --gnn_eval_interval 5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}_layer_3
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 2 \
    --lm_type $lm_model_type \
    --gnn_batch_size 512 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 150 \
    --gnn_dropout 0.2 \
    --gnn_lr 0.0005 \
    --gnn_num_layers 3 \
    --gnn_weight_decay 2e-6 \
    --gnn_eval_warmup 10 \
    --gnn_eval_interval 5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
