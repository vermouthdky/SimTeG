dataset=ogbn-arxiv-tape
model_type=all-roberta-large-v1
suffix=main

bash scripts/train.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo sentence-transformers/${model_type} \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --batch_size 20 \
    --eval_batch_size 200 \
    --accum_interval 5 \
    --label_smoothing 0.3 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --peft_r 4 \
    --peft_lora_alpha 8 \
    --peft_lora_dropout 0.3 \
    --header_dropout_prob 0.6 \
    --use_peft \
    --deepspeed ds_config.json
