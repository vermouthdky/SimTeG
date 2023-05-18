dataset=$1
lm_model_type=$2
suffix=optuna

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
    --epochs 10 \
    --warmup_ratio 0.15 \
    --lr_scheduler_type linear \
    --n_trials 20
