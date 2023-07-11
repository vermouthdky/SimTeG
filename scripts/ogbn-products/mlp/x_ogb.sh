dataset=ogbn-products
model_type=MLP

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_ogb
bash scripts/single_gpu_train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --single_gpu 3 \
    --n_exps 10 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.5 \
    --gnn_label_smoothing 0.3 \
    --gnn_lr 0.002 \
    --gnn_weight_decay 1e-4
