dataset=ogbl-citation2
model_type=minilm-l6
suffix=finetune

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 10 \
    --eval_batch_size 10 \
    --accum_interval 5 \
    --eval_patience 100000 \
    --eval_delay 10 \
    --epochs 2 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --deepspeed ds_config.json \
    --fp16
