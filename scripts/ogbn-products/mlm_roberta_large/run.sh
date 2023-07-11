# bash scripts/ogbn-products/mlm_roberta_large/main.sh
bash scripts/ogbn-products/mlm_roberta_large/fix.sh

model_type=GraphSAGE
dataset=ogbn-products
lm_model_type=roberta-large
suffix=main_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 0 \
    --lm_type $lm_model_type \
    --gnn_batch_size 1024 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 100 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.3 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 1e-5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=roberta-large
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 1 \
    --lm_type $lm_model_type \
    --gnn_batch_size 1024 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 100 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.3 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 1e-5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

model_type=MLP
lm_model_type=roberta-large
suffix=main_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/main/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 2 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.5 \
    --gnn_label_smoothing 0.3 \
    --gnn_lr 0.002 \
    --gnn_weight_decay 1e-4 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=roberta-large
suffix=main_X_${lm_model_type}_fix
bert_x_dir=out/${dataset}/${lm_model_type}/fix/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 5 \
    --single_gpu 3 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.5 \
    --gnn_label_smoothing 0.3 \
    --gnn_lr 0.002 \
    --gnn_weight_decay 1e-4 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &
