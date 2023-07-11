dataset=ogbn-arxiv
model_type=all-roberta-large-v1
suffix=main

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 10 \
    --eval_batch_size 100 \
    --accum_interval 5 \
    --label_smoothing 0.3 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear
