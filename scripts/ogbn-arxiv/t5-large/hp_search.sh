dataset=ogbn-arxiv
model_type=deberta-v2-xxlarge
suffix=optuna_peft

bash scripts/optuna.sh --model_type $model_type --dataset $dataset --suffix $suffix \
    --pretrained_repo microsoft/${model_type} \
    --lr 1e-5 \
    --weight_decay 1e-5 \
    --batch_size 5 \
    --eval_batch_size 50 \
    --accum_interval 10 \
    --label_smoothing 0.3 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --use_peft \
    --fp16 \
    --deepspeed ds_config.json \
    --n_trials 10
