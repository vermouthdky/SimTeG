dataset=ogbn-arxiv
model_type=instructor-xl
suffix=main

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo hkunlp/instructor-xl \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --batch_size 3 \
    --eval_batch_size 30 \
    --accum_interval 10 \
    --label_smoothing 0.3 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --use_peft \
    --bf16 \
    --deepspeed ds_config.json
