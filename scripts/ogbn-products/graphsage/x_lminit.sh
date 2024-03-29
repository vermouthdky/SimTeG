dataset=ogbn-products
model_type=GraphSAGE

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 0 \
    --lm_type $lm_model_type \
    --gnn_batch_size 1024 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 100 \
    --gnn_dropout 0.15 \
    --gnn_label_smoothing 0.1 \
    --gnn_lr 0.01 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 2e-6 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 1 \
    --lm_type $lm_model_type \
    --gnn_batch_size 1024 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 100 \
    --gnn_dropout 0.6 \
    --gnn_label_smoothing 0.2 \
    --gnn_lr 0.001 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 5e-5 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir &

lm_model_type=e5-large
suffix=main_X_${lm_model_type}
bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --single_gpu 2 \
    --lm_type $lm_model_type \
    --gnn_batch_size 1024 \
    --gnn_eval_batch_size 4096 \
    --gnn_epochs 100 \
    --gnn_dropout 0.4 \
    --gnn_label_smoothing 0.4 \
    --gnn_lr 0.006 \
    --gnn_num_layers 2 \
    --gnn_weight_decay 5e-7 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
