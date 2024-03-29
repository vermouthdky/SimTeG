dataset=ogbn-products
model_type=MLP

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_${lm_model_type}_peft
bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --single_gpu 0 \
    --n_exps 10 \
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

lm_model_type=all-roberta-large-v1
suffix=main_X_${lm_model_type}_peft
bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --single_gpu 1 \
    --n_exps 10 \
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

lm_model_type=e5-large
suffix=main_X_${lm_model_type}_peft
bert_x_dir=out/${dataset}/${lm_model_type}/optuna_peft/best/cached_embs/x_embs.pt
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --single_gpu 2 \
    --n_exps 10 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.5 \
    --gnn_label_smoothing 0.3 \
    --gnn_lr 0.002 \
    --gnn_weight_decay 1e-4 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
