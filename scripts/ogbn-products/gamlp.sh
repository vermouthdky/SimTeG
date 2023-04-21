dataset=ogbn-products
model_type=GAMLP
suffix=x_ogb
bert_x_dir=out/ogbn-products/Deberta/main/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --gnn_lr 1e-2 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-4 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.7 \
    --gnn_num_layers 4 \
    --lr_scheduler_type linear \
    --gnn_label_smoothing 0.2
# --use_bert_x \
# --bert_x_dir $bert_x_dir
