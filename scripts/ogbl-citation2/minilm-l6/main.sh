dataset=ogbl-citation2
model_type=all-MiniLM-L6-v2
suffix=main

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 5 \
    --epochs 2 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --deepspeed ds_config.json \
    --use_peft
