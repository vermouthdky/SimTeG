dataset=$1
lm_model_type=$2
suffix=optuna_peft

large_model_types=("e5-large" "all-roberta-large-v1")
if [[ " ${large_model_types[*]} " == *" ${lm_model_type} "* ]]; then
    batch_size=10
    eval_batch_size=100
else
    batch_size=20
    eval_batch_size=200
fi

bash scripts/optuna.sh --model_type $lm_model_type --dataset $dataset --suffix $suffix \
    --batch_size $batch_size \
    --eval_batch_size $eval_batch_size \
    --lr 5e-5 \
    --weight_decay 1e-5 \
    --accum_interval 5 \
    --label_smoothing 0.3 \
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --use_peft \
    --n_trials 10
