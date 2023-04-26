dataset=ogbn-arxiv
model_type=all-mpnet-base-v2
suffix=main

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 5 \
    --label_smoothing 0.32 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear

lm_type=${model_type}
model_type=GAMLP
suffix=main
bert_x_dir=out/ogbn-arxiv/${lm_type}/main/cached_embs/iter_0_x_embs.pt

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --lm_type $lm_type \
    --gnn_lr 5e-3 \
    --gnn_eval_interval 5 \
    --gnn_weight_decay 2e-4 \
    --gnn_batch_size 10000 \
    --gnn_eval_batch_size 10000 \
    --gnn_epochs 100 \
    --gnn_dropout 0.7 \
    --gnn_num_layers 4 \
    --lr_scheduler_type constant \
    --gnn_label_smoothing 0.2 \
    --use_bert_x \
    --bert_x_dir $bert_x_dir
