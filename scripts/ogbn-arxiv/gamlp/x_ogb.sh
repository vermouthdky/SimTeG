dataset=ogbn-arxiv
model_type=GAMLP

lm_model_type=all-MiniLM-L6-v2
suffix=main_X_ogb
bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --n_exps 10 \
    --lm_type $lm_model_type \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.37 \
    --gnn_lr 0.007 \
    --gnn_num_layers 5 \
    --gnn_warmup_ratio 0.41 \
    --gnn_weight_decay 5e-6
